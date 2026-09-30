"""Replay actual final500 updates without changing any selected benchmark weight."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from geopolicy.io import save_json
from job_runtime import stop_owned_job
from resource_watch import ResourceWatch

root = Path("artifacts/full_training_resume")
root.mkdir(parents=True, exist_ok=True)
reference = Path("artifacts/main_runs/mono_s1")
selected = torch.load(reference / "best.pt", map_location="cpu", weights_only=False)
start = selected["progress"]["update"]
assert start == 7500 and selected["config"]["updates"] == 8000
del selected
command = [
    sys.executable,
    "-m",
    "geopolicy.cli",
    "student-train",
    "--mode",
    "mono",
    "--seed",
    "1",
    "--updates",
    "8000",
    "--limit",
    "200",
    "--batch-size",
    "32",
    "--device",
    "cuda",
    "--resume",
    str(reference / "best.pt"),
    "--out",
    str(root),
]
watch = ResourceWatch(root)
with (root / "execution.log").open("w") as log:
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    begun = time.monotonic()
    try:
        while proc.poll() is None:
            watch.sample()
            if time.monotonic() - begun > 600:
                raise TimeoutError("Resume check exceeded600s")
            time.sleep(1)
    except BaseException:
        stop_owned_job(proc)
        raise
    if proc.returncode:
        raise RuntimeError("Resume training failed; see disposable execution log")
expected = torch.load(reference / "latest.pt", map_location="cpu", weights_only=False)
actual = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)


def exact(x, y):
    if isinstance(x, torch.Tensor):
        return torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(exact(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(exact(a, b) for a, b in zip(x, y))
    return x == y


checks = {
    key: exact(expected[key], actual[key])
    for key in [
        "model",
        "optimizer",
        "scheduler",
        "normalization",
        "config",
        "progress",
        "random",
        "extra",
    ]
}
maximum = max(
    float((expected["model"][k] - actual["model"][k]).abs().max()) for k in expected["model"]
)
report = {
    "source_checkpoint": str(reference / "best.pt"),
    "reference_final": str(reference / "latest.pt"),
    "disposable_replay": str(root / "latest.pt"),
    "from_update": start,
    "to_update": 8000,
    "actual_replayed_optimizer_updates": 8000 - start,
    "device": "cuda",
    "seed": 1,
    "mode": "mono",
    "same200_training_demonstrations": True,
    "exact_checks": checks,
    "maximum_raw_model_weight_difference": maximum,
    "all_exact": all(checks.values()),
    "selected_benchmark_checkpoints_modified": False,
}
save_json("results/full_training_resume.json", report)
print(json.dumps(report, indent=2))
if not report["all_exact"]:
    raise RuntimeError(
        "Actual training resume differs; inspect preserved original/replay artifacts"
    )
