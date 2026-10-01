"""Fetch the compact release bundle, verify its identity, and safely extract it."""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_bundle(archive=None, out=None):
    manifest = json.loads((ROOT / "docs/releases/v2_demo.json").read_text(encoding="utf8"))
    archive = Path(archive) if archive else ROOT / "artifacts/v2/demo_bundle.zip"
    out = Path(out) if out else ROOT / "artifacts/v2/demo"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with tempfile.TemporaryDirectory(dir=archive.parent) as temporary:
            downloaded = Path(temporary) / "bundle.zip"
            request = urllib.request.Request(
                manifest["download_url"], headers={"User-Agent": "GeoPolicy-Bench-demo"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with downloaded.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            verify_archive(downloaded, manifest)
            shutil.move(str(downloaded), archive)
    verify_archive(archive, manifest)
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["members_sha256"]):
            raise ValueError("Unexpected or duplicate bundle members")
        for name in names:
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Only flat, declared bundle paths are allowed")
            content = zipped.read(name)
            if hashlib.sha256(content).hexdigest() != manifest["members_sha256"][name]:
                raise ValueError(f"Bundle member checksum mismatch: {name}")
            target = out / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"Existing file differs; choose a new output folder: {target}")
        out.mkdir(parents=True, exist_ok=True)
        zipped.extractall(out)
    print(f"Verified compact checkpoint bundle ready: {out}")
    return out


def verify_archive(path, manifest):
    if path.stat().st_size != manifest["size_bytes"]:
        raise ValueError("Demo archive size does not match the published manifest")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Demo archive SHA-256 does not match the published manifest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Use a previously downloaded local ZIP")
    parser.add_argument("--out", help="Extract into a separate demo folder")
    args = parser.parse_args()
    prepare_bundle(args.archive, args.out)
