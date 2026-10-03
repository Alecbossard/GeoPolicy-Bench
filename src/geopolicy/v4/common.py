import hashlib
import json
import os
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
    os.replace(tmp, path)


def plan():
    return read("configs/v4/plan.json")
