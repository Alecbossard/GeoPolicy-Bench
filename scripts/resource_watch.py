"""Lightweight process-local resource guard for long VLA inference."""

import ctypes
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from geopolicy.io import save_json


class ResourceWatch:
    def __init__(self, out, interval=15):
        self.out = Path(out)
        self.interval = interval
        self.previous = 0
        self.hot = 0
        self.low_commit = 0
        self.rows = []

    def sample(self):
        if time.monotonic() - self.previous < self.interval:
            return
        self.previous = time.monotonic()
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        fields = gpu.stdout.strip().split(",")
        temp = float(fields[2]) if gpu.returncode == 0 and len(fields) == 3 else None
        commit = None
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong)
                    for name in [
                        "total_physical",
                        "free_physical",
                        "total_commit",
                        "free_commit",
                        "total_virtual",
                        "free_virtual",
                        "extended_virtual",
                    ]
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                commit = status.free_commit
        disk = shutil.disk_usage(Path.cwd().anchor).free
        self.hot = self.hot + 1 if temp is not None and temp >= 90 else 0
        self.low_commit = (
            self.low_commit + 1 if commit is not None and commit < 512 * 1024**2 else 0
        )
        row = {
            "time": time.time(),
            "gpu_used_free_mib_temperature_c": gpu.stdout.strip(),
            "free_commit_bytes": commit,
            "free_disk_bytes": disk,
        }
        self.rows.append(row)
        save_json(self.out / "resources.json", self.rows)
        if disk < 12 * 1024**3 or self.hot >= 3 or self.low_commit >= 3:
            save_json(self.out / "resource_stop.json", row)
            raise RuntimeError(
                "VLA resource guard stopped this job; completed episode rows are preserved"
            )
