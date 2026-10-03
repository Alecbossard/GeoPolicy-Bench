import hashlib
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path).resolve()
    assert any(
        path.relative_to(ROOT).as_posix().startswith(p + "/v4/")
        for p in ("configs", "results", "artifacts", "docs")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    # Windows readers can briefly deny replacement without corrupting the old file.
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, None):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def runtime_identity():
    return dict(
        python_version=list(sys.version_info[:3]),
        packages={
            k: metadata.version(k)
            for k in ("torch", "numpy", "mujoco", "robosuite", "h5py")
        },
    )


def verify_runtime(frozen):
    current = runtime_identity()
    assert (
        current["python_version"] == frozen["python_version"]
    ), "Python version changed"
    assert current["packages"] == frozen["packages"], "Package versions changed"
    return current


def plan():
    return read("configs/v4/plan.json")
