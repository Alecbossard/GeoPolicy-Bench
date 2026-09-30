"""Predefined single-seed, controlled data-count and view-dropout ablations."""

import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
from job_runtime import stop_owned_job

root = Path("artifacts/secondary_runs")
root.mkdir(parents=True, exist_ok=True)
resources = []
for name, limit, dropout in [("fusion_50d_s0", 50, 0), ("fusion_dropout_s0", 200, 0.2)]:
    out = root / name
    if (out / "manifest.json").exists():
        continue
    cmd = [
        sys.executable,
        "-m",
        "geopolicy.cli",
        "student-train",
        "--mode",
        "fusion",
        "--seed",
        "0",
        "--updates",
        "8000",
        "--out",
        str(out),
        "--limit",
        str(limit),
        "--batch-size",
        "32",
        "--device",
        "cuda",
        "--view-dropout",
        str(dropout),
    ]
    if (out / "latest.pt").exists():
        cmd += ["--resume", str(out / "latest.pt")]
    with (root / f"{name}.log").open("a") as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        hot = 0
        start = time.time()
        print("Training", name, proc.pid, flush=True)
        while proc.poll() is None:
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            values = gpu.stdout.strip().split(",")
            temperature = float(values[0]) if len(values) == 3 else 0
            disk = psutil.disk_usage(str(Path.cwd().anchor))
            mem = psutil.virtual_memory()
            row = {
                "time": time.time(),
                "run": name,
                "gpu": gpu.stdout.strip(),
                "temperature_c": temperature,
                "ram_available_bytes": mem.available,
                "disk_free_bytes": disk.free,
            }
            resources.append(row)
            (root / "resources.json").write_text(json.dumps(resources, indent=2))
            hot = hot + 1 if temperature >= 90 else 0
            if hot >= 3 or disk.free < 12 * 1024**3 or time.time() - start > 3600:
                stop_owned_job(proc)
                (root / "resource_stop.json").write_text(json.dumps(row, indent=2))
                raise RuntimeError("Resource guard")
            time.sleep(15)
    if proc.returncode:
        raise RuntimeError(f"Training failed: {name}")
    print("Finished", name, flush=True)
