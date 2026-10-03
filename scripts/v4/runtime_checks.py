"""Exercise real Windows replacement locks and runtime-version rejection."""

import copy
import ctypes
from ctypes import wintypes
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from geopolicy.v4.common import runtime_identity, verify_runtime, write


def main():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    path = ROOT / "artifacts/v4/checks/atomic.json"
    write(path, {"value": "old"})

    def lock():
        handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
        assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
        return handle

    handle = lock()
    closer = threading.Thread(
        target=lambda: (time.sleep(0.12), kernel.CloseHandle(handle))
    )
    closer.start()
    started = time.perf_counter()
    write(path, {"value": "recovered"})
    elapsed = time.perf_counter() - started
    closer.join()
    assert json.loads(path.read_text())["value"] == "recovered" and elapsed >= 0.12

    handle = lock()
    persistent_rejected = False
    try:
        write(path, {"value": "pending"})
    except PermissionError:
        persistent_rejected = True
    finally:
        kernel.CloseHandle(handle)
    assert persistent_rejected and json.loads(path.read_text())["value"] == "recovered"
    assert path.with_suffix(".json.tmp").exists()
    write(path, {"value": "final"})

    current = runtime_identity()
    assert verify_runtime(current) == current
    rejected = []
    for key in ("python_version", "packages"):
        wrong = copy.deepcopy(current)
        if key == "python_version":
            wrong[key][2] += 1
        else:
            wrong[key]["torch"] = "0.0.invalid"
        try:
            verify_runtime(wrong)
        except AssertionError:
            rejected.append(key)
    assert rejected == ["python_version", "packages"]
    result = dict(
        verified=True,
        transient_real_windows_lock_recovered=True,
        recovery_seconds=elapsed,
        persistent_lock_rejected_old_json_preserved=True,
        pending_temp_preserved=True,
        runtime=current,
        runtime_changes_rejected=rejected,
    )
    write(ROOT / "results/v4/io_runtime_checks.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
