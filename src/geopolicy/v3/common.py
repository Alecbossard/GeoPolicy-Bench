import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    assert any(
        relative.startswith(p)
        for p in ("artifacts/v3/", "results/v3/", "configs/v3/", "docs/v3/")
    ), f"V3 write outside its namespace: {relative}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    os.replace(tmp, path)


def preservation(create=False):
    target = ROOT / "configs/v3/preservation.json"
    if create:
        assert not target.exists(), "Preservation identity already exists"
        tracked = (
            subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
            .decode()
            .split("\0")
        )
        files = {ROOT / p for p in tracked if p}
        # Every historical data/weight/result artifact, including complete traces.
        # Exclude presentation clones (copies), isolated replay copies, and new V3.
        for path in (ROOT / "artifacts").rglob("*"):
            if path.is_file() and path.suffix.lower() in (
                ".h5",
                ".pt",
                ".zip",
                ".json",
                ".csv",
                ".npz",
            ):
                rel = path.relative_to(ROOT / "artifacts").as_posix()
                if not rel.startswith(
                    ("v3/", "github_presentation/", "v2/portable_demo/")
                ):
                    files.add(path)
        manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
        write(
            target,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "base_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT
                )
                .decode()
                .strip(),
                "files": manifest,
            },
        )
    manifest = json.loads(target.read_text())
    failures = [
        p
        for p, expected in manifest["files"].items()
        if not (ROOT / p).exists() or sha(ROOT / p) != expected
    ]
    result = {
        "checked_files": len(manifest["files"]),
        "failures": failures,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }
    write(ROOT / "results/v3/preservation_check.json", result)
    assert not failures, failures
    print(json.dumps(result), flush=True)
    return result
