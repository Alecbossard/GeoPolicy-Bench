"""Serial, resumable final test. First20 per cell, then extend primary cells to100."""

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
from job_runtime import stop_owned_job

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["main", "counterfactual"], default="main")
a = p.parse_args()
protocol = json.loads(Path("configs/benchmark_protocol.json").read_text())
assert protocol["frozen"]
root = Path(
    "artifacts/final_evaluations" if a.stage == "main" else "artifacts/counterfactual_evaluations"
)
root.mkdir(parents=True, exist_ok=True)
jobs = []
for mode in ["fusion", "mono", "act"]:
    for seed in [0, 1, 2]:
        assert (
            Path("artifacts/main_runs") / f"{mode}_s{seed}" / "manifest.json"
        ).exists(), "Complete all main training before final test"
        checkpoint = Path("artifacts/main_runs") / f"{mode}_s{seed}" / "best.pt"
        if a.stage == "main":
            for condition in protocol["evaluation"]["conditions"]:
                target = (
                    protocol["evaluation"]["main_episodes_per_condition"][condition]
                    if mode != "act"
                    else 20
                )
                # Same directory/identity: second call skips the completed first20.
                for count in sorted(set([20, target])):
                    out = root / f"{mode}_s{seed}" / condition
                    jobs.append((mode, seed, condition, count, out, checkpoint, []))
        elif mode != "act":
            for object_id in [0, 1]:
                for goal_id in [0, 1]:
                    out = root / f"{mode}_s{seed}" / f"object{object_id}_goal{goal_id}"
                    jobs.append(
                        (
                            mode,
                            seed,
                            "nominal",
                            10,
                            out,
                            checkpoint,
                            ["--object-id", str(object_id), "--goal-id", str(goal_id)],
                        )
                    )
resources = []
for mode, seed, condition, count, out, checkpoint, extra in jobs:
    rows_path = out / "rollouts.json"
    if rows_path.exists() and len(json.loads(rows_path.read_text())) >= count:
        print("Complete", str(out), count, flush=True)
        continue
    first_seed = 200000 if a.stage == "main" else 200100
    cmd = [
        sys.executable,
        "-m",
        "geopolicy.cli",
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out),
        "--episodes",
        str(count),
        "--first-seed",
        str(first_seed),
        "--conditions",
        condition,
        "--device",
        "cpu",
        "--execute-steps",
        "2",
        "--videos",
        "1",
    ] + extra
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with (out / "execution.log").open("a") as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        print("Evaluating", mode, seed, condition, count, str(out), "PID", proc.pid, flush=True)
        hot = 0
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
            fields = gpu.stdout.strip().split(",")
            temperature = float(fields[0]) if len(fields) == 3 else None
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(str(Path.cwd().anchor))
            sample = {
                "time": time.time(),
                "mode": mode,
                "seed": seed,
                "condition": condition,
                "temperature_c": temperature,
                "gpu": gpu.stdout.strip(),
                "ram_available_bytes": mem.available,
                "disk_free_bytes": disk.free,
                "pid": proc.pid,
            }
            resources.append(sample)
            (root / "resources.json").write_text(json.dumps(resources, indent=2))
            hot = hot + 1 if temperature is not None and temperature >= 90 else 0
            if disk.free < 12 * 1024**3 or hot >= 3 or time.time() - start > 3600:
                stop_owned_job(proc)
                (root / "resource_stop.json").write_text(json.dumps(sample, indent=2))
                raise RuntimeError("Resource guard; completed episodes preserved")
            time.sleep(15)
    if proc.returncode:
        raise RuntimeError(f"Evaluation failed: {out/'execution.log'}")
    rows = json.loads(rows_path.read_text())
    rate = sum(r["success"] for r in rows) / len(rows)
    print(
        "Finished",
        mode,
        seed,
        condition,
        len(rows),
        "success",
        rate,
        "wall",
        round(time.time() - start, 1),
        flush=True,
    )
