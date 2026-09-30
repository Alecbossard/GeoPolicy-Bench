"""Serial GPU experiments, checkpoints/logs and conservative resource guard."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
from job_runtime import stop_owned_job

p = argparse.ArgumentParser()
p.add_argument("--updates", type=int, default=8000)
p.add_argument("--modes", nargs="+", default=["fusion", "mono", "act"])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
a = p.parse_args()
root = Path("artifacts/main_runs")
root.mkdir(parents=True, exist_ok=True)
protocol = json.loads(Path("configs/benchmark_protocol.json").read_text())
assert protocol["frozen"] and a.updates == protocol["training"]["updates"]
resources = []
manifest = []
for mode in a.modes:
    for seed in a.seeds:
        out = root / f"{mode}_s{seed}"
        if (out / "manifest.json").exists():
            m = json.loads((out / "manifest.json").read_text())
            if m["config"]["updates"] == a.updates:
                print("Already complete", mode, seed, flush=True)
                continue
        cmd = [
            sys.executable,
            "-m",
            "geopolicy.cli",
            "student-train",
            "--mode",
            mode,
            "--seed",
            str(seed),
            "--updates",
            str(a.updates),
            "--out",
            str(out),
            "--limit",
            "200",
            "--batch-size",
            "32",
            "--device",
            "cuda",
        ]
        if (out / "latest.pt").exists():
            cmd.extend(["--resume", str(out / "latest.pt")])
        log = root / f"{mode}_s{seed}.log"
        start = time.time()
        bad_temperature = 0
        bad_commit = 0
        with log.open("a") as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
            print("Training", mode, seed, "PID", proc.pid, "log", str(log), flush=True)
            while proc.poll() is None:
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                fields = query.stdout.strip().split(",")
                temperature = float(fields[0]) if len(fields) == 3 else None
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage(str(Path.cwd().anchor))
                sample = {
                    "time": time.time(),
                    "mode": mode,
                    "seed": seed,
                    "temperature_c": temperature,
                    "gpu": query.stdout.strip(),
                    "ram_available_bytes": mem.available,
                    "disk_free_bytes": disk.free,
                    "pid": proc.pid,
                }
                # Windows commit availability is read via native GlobalMemoryStatusEx.
                if sys.platform == "win32":
                    import ctypes

                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    status = MEMORYSTATUSEX()
                    status.dwLength = ctypes.sizeof(status)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                        sample["commit_available_bytes"] = status.ullAvailPageFile
                        bad_commit = (
                            bad_commit + 1 if status.ullAvailPageFile < 512 * 1024**2 else 0
                        )
                bad_temperature = (
                    bad_temperature + 1 if temperature is not None and temperature >= 90 else 0
                )
                resources.append(sample)
                (root / "resources.json").write_text(json.dumps(resources, indent=2))
                if (
                    disk.free < 12 * 1024**3
                    or bad_temperature >= 3
                    or bad_commit >= 3
                    or time.time() - start > 3600
                ):
                    stop_owned_job(proc)
                    sample["stopped_for_resource_guard"] = True
                    (root / "resource_stop.json").write_text(json.dumps(sample, indent=2))
                    raise RuntimeError(
                        "Resource guard stopped current job; latest periodic checkpoint preserved. Review resource_stop.json."
                    )
                time.sleep(15)
        if proc.returncode:
            raise RuntimeError(f"Training failed {mode}/{seed}: {log}")
        manifest.append(
            {
                "mode": mode,
                "seed": seed,
                "wall_seconds": time.time() - start,
                "log": str(log),
                "out": str(out),
            }
        )
        (root / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2))
        print("Finished", mode, seed, flush=True)
