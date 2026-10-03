"""Package the verified local portable demo, without any publication."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    bundle = ROOT / "artifacts/v3/portable_demo"
    result = json.loads((bundle / "replay/verification.json").read_text())
    assert (
        result["exact_frozen_trace"]
        and result["physical_success"]
        and result["strict_v2_success"]
    )
    assert (
        Path(result["imported_source_directory"]).resolve()
        == (bundle / "src/geopolicy").resolve()
    )
    paths = [
        p
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and not p.relative_to(bundle).as_posix().startswith("artifacts/")
    ]
    target = ROOT / "artifacts/v3/demo_bundle.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    metadata = dict(
        path=target.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        bytes=target.stat().st_size,
        files=len(paths),
        portable_replay_verified=True,
        published=False,
    )
    (ROOT / "results/v3/demo_bundle.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
