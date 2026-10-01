"""Read-only laptop resource monitoring with configured limits."""

import ctypes
import os
import shutil
import subprocess
import time
import psutil
from geopolicy.io import save_json


class ResourceWatch:
    def __init__(self, out, limits):
        self.out, self.limits = out, limits
        self.previous = 0
        self.hot = self.low_commit = 0
        self.rows = []

    def sample(self, force=False):
        if not force and time.monotonic() - self.previous < self.limits["sample_interval_seconds"]:
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
        values = [float(v) for v in gpu.stdout.strip().split(",")] if gpu.returncode == 0 else None
        commit = None
        if os.name == "nt":

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong)
                    for n in [
                        "physical_total",
                        "physical_free",
                        "commit_total",
                        "commit_free",
                        "virtual_total",
                        "virtual_free",
                        "extended_virtual",
                    ]
                ]

            s = Status()
            s.length = ctypes.sizeof(s)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                commit = s.commit_free
        memory = psutil.Process().memory_info()
        disk = shutil.disk_usage(".").free
        row = {
            "time": time.time(),
            "gpu_used_free_mib_temperature_c": values,
            "free_commit_bytes": commit,
            "free_disk_bytes": disk,
            "available_ram_bytes": psutil.virtual_memory().available,
            "rss_bytes": memory.rss,
            "private_bytes": getattr(memory, "private", None),
        }
        self.rows.append(row)
        save_json(str(self.out) + "/resources.json", self.rows)
        self.hot = (
            self.hot + 1 if values and values[2] >= self.limits["maximum_temperature_c"] else 0
        )
        self.low_commit = (
            self.low_commit + 1
            if commit is not None and commit < self.limits["minimum_free_commit_mib"] * 1024**2
            else 0
        )
        if (
            disk < self.limits["minimum_free_disk_gib"] * 1024**3
            or self.hot >= self.limits["consecutive_unsafe_samples"]
            or self.low_commit >= self.limits["consecutive_unsafe_samples"]
        ):
            save_json(str(self.out) + "/resource_stop.json", row)
            raise RuntimeError("Configured laptop resource limit reached; saved progress preserved")
        return row
