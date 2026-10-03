"""Lightweight startup guard, before importing the CUDA-enabled Torch runtime."""

import ctypes
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from .common import ROOT, write


def check_readiness(snapshot, runtime_limits, in_job_limits):
    problems = []
    commit = snapshot["free_commit_bytes"]
    if (
        commit is not None
        and commit
        < runtime_limits["minimum_free_commit_before_heavy_job_gib"] * 1024**3
    ):
        problems.append("insufficient Windows free commit before startup")
    if snapshot["free_disk_bytes"] < in_job_limits["minimum_free_disk_gib"] * 1024**3:
        problems.append("insufficient free disk space")
    if snapshot["gpu_free_mib"] < runtime_limits["minimum_free_gpu_mib"]:
        problems.append("insufficient free GPU memory")
    if snapshot["gpu_temperature_c"] >= in_job_limits["maximum_temperature_c"]:
        problems.append("GPU temperature exceeds configured limit")
    return problems


def preflight(command):
    commit = None
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "physical_total",
                    "physical_free",
                    "commit_total",
                    "commit_free",
                    "virtual_total",
                    "virtual_free",
                    "extended_virtual",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError(
                "Cannot read Windows memory status; heavy job not started"
            )
        commit = status.commit_free
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if gpu.returncode:
        raise RuntimeError("Cannot read GPU resources; heavy job not started")
    gpu_free, temperature = [float(v) for v in gpu.stdout.strip().split(",")]
    runtime_limits = json.loads((ROOT / "configs/v3/runtime_limits.json").read_text())
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    snapshot = dict(
        command=command,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        free_commit_bytes=commit,
        free_disk_bytes=shutil.disk_usage(ROOT).free,
        gpu_free_mib=gpu_free,
        gpu_temperature_c=temperature,
    )
    problems = check_readiness(snapshot, runtime_limits, plan["resource_limits"])
    snapshot.update(
        problems=problems, heavy_job_started=False, runtime_limits=runtime_limits
    )
    write(ROOT / "artifacts/v3/resource_preflight.json", snapshot)
    if problems:
        print(json.dumps(snapshot), flush=True)
        raise RuntimeError("Heavy V3 job not started: " + "; ".join(problems))
    return snapshot
