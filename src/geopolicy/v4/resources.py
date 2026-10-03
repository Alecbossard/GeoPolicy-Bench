"""V4-only outputs; lightweight guard executes before importing Torch."""

import ctypes
import os
import shutil
import subprocess
from datetime import datetime, timezone
from .common import ROOT, plan, write


def preflight(command):
    limits = plan()
    commit = None
    if os.name == "nt":

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong)
                for n in (
                    "physical_total",
                    "physical_free",
                    "commit_total",
                    "commit_free",
                    "virtual_total",
                    "virtual_free",
                    "extended_virtual",
                )
            ]

        s = Status()
        s.length = ctypes.sizeof(s)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        commit = s.commit_free
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    free, temp = map(float, gpu.stdout.strip().split(","))
    problems = []
    if (
        commit is not None
        and commit
        < limits["runtime"]["minimum_free_commit_before_heavy_job_gib"] * 1024**3
    ):
        problems.append("insufficient free Windows commit before startup")
    disk = shutil.disk_usage(ROOT).free
    if disk < limits["resource_limits"]["minimum_free_disk_gib"] * 1024**3:
        problems.append("insufficient disk")
    if free < limits["runtime"]["minimum_free_gpu_mib"]:
        problems.append("insufficient VRAM")
    if temp >= limits["resource_limits"]["maximum_temperature_c"]:
        problems.append("GPU temperature")
    record = dict(
        command=command,
        utc=datetime.now(timezone.utc).isoformat(),
        free_commit_bytes=commit,
        free_disk_bytes=disk,
        gpu_free_mib=free,
        gpu_temperature_c=temp,
        problems=problems,
    )
    write(ROOT / "artifacts/v4/resource_preflight.json", record)
    if problems:
        raise RuntimeError(str(record))
    return record
