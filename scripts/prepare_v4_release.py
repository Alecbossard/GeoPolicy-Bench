"""Prepare/check a local V4 release; never train or rewrite study outputs."""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/releases/v4-presentation"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def dump(path, value):
    assert path.is_relative_to(OUT) or path.is_relative_to(ROOT / "docs/releases")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


def snapshot():
    target = OUT / "preservation.json"
    assert not target.exists(), "Reuse the first snapshot; do not reset preservation"
    inventory = {}
    for folder in ("src", "configs", "results", "artifacts"):
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or path.is_relative_to(OUT):
                continue
            if "__pycache__" in path.parts or path.suffix == ".tmp":
                continue
            inventory[path.relative_to(ROOT).as_posix()] = sha(path)
    docs = [
        "README.md",
        "docs/v3/README.md",
        "docs/v4/README.md",
        "docs/v4/report.md",
        "docs/v4/reproduction.md",
        "docs/v4/PROGRESS.md",
        "scripts/v4/report.py",
    ]
    backups = {}
    for rel in docs:
        dst = OUT / "history" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
        backups[rel] = sha(dst)
    historical_readme = ROOT / "docs/history/README_V2_original.txt"
    historical_readme.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "README.md", historical_readme)
    dump(
        target,
        dict(
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            protected_files=inventory,
            documentation_backups=backups,
            no_results_or_frozen_protocol_changes_allowed=True,
        ),
    )
    print(json.dumps(dict(protected_files=len(inventory), snapshot=str(target))))


def build():
    assert (OUT / "preservation.json").exists()
    assets_dir = OUT / "assets"
    assert not assets_dir.exists(), "Release assets already built; never overwrite them"
    assets_dir.mkdir(parents=True)
    source_manifest = read(ROOT / "artifacts/v4/demo/manifest.json")
    export = read(ROOT / "results/v4/demo_export.json")
    originals = {
        "geopolicy-v4-demo.zip": "artifacts/v4/demo_bundle.zip",
        "checkpoint.pt": "artifacts/v4/demo/checkpoint.pt",
        "demo.mp4": "artifacts/v4/demo/reproduced/demo.mp4",
        "demo.gif": "artifacts/v4/demo/reproduced/demo.gif",
        "requirements-lock.txt": "requirements-lock.txt",
        "checkpoint_manifest.json": "artifacts/v4/demo/manifest.json",
    }
    assert sha(ROOT / originals["geopolicy-v4-demo.zip"]) == export["bundle_sha256"]
    assert sha(ROOT / originals["checkpoint.pt"]) == source_manifest["compact_sha256"]
    for name, rel in originals.items():
        shutil.copy2(ROOT / rel, assets_dir / name)
    runtime = {key: source_manifest[key] for key in ("python_version", "packages")}
    dump(assets_dir / "runtime.json", runtime)
    for name in ("demo.mp4", "demo.gif"):
        shutil.copy2(assets_dir / name, ROOT / "docs/media" / ("v4_" + name))
    shutil.copy2(
        ROOT / "artifacts/v3/demo/replay/demo.mp4", ROOT / "docs/media/v3_demo.mp4"
    )
    assets = {
        path.name: dict(sha256=sha(path), size_bytes=path.stat().st_size)
        for path in sorted(assets_dir.iterdir())
    }
    media = {
        f"docs/media/{name}": dict(
            sha256=sha(ROOT / f"docs/media/{name}"),
            size_bytes=(ROOT / f"docs/media/{name}").stat().st_size,
        )
        for name in ("v4_demo.mp4", "v4_demo.gif", "v3_demo.mp4")
    }
    with zipfile.ZipFile(assets_dir / "geopolicy-v4-demo.zip") as archive:
        member_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if not name.endswith("/")
        }
    manifest = dict(
        status="prepared_locally_not_published",
        release_tag="v4-demo",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        assets=assets,
        media=media,
        runtime=runtime,
        members_sha256=member_hashes,
        original_bundle_sha256=export["bundle_sha256"],
        checkpoint_sha256=source_manifest["compact_sha256"],
        demo_policy="fusion_aug_s0",
        demo_scene=500000,
        demo_condition="absent",
        demo_rule="First reserved scene; policy and condition selected before test access",
        result_policy="fusion_clean",
        contains_training_data=False,
        contains_optimizer=False,
        archive_member_count=len(member_hashes),
        publication_url=None,
    )
    dump(ROOT / "docs/releases/v4_demo.json", manifest)
    dump(assets_dir / "release_manifest.json", manifest)
    checksum_names = sorted(p.name for p in assets_dir.iterdir())
    (assets_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha(assets_dir / name)}  {name}\n" for name in checksum_names),
        encoding="ascii",
    )
    print(
        json.dumps(
            dict(
                assets=len(checksum_names),
                archive_bytes=assets["geopolicy-v4-demo.zip"]["size_bytes"],
                bundle_sha256=manifest["original_bundle_sha256"],
            )
        )
    )


def replay():
    manifest = read(ROOT / "docs/releases/v4_demo.json")
    archive_path = OUT / "assets/geopolicy-v4-demo.zip"
    assert sha(archive_path) == manifest["original_bundle_sha256"]
    isolated = OUT / "isolated_demo"
    assert (
        not isolated.exists()
    ), "Keep the isolated replay evidence; use a new delivery identity"
    isolated.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (isolated / member.filename).resolve()
            assert destination.is_relative_to(isolated.resolve()), member.filename
            assert not (member.external_attr >> 16) & 0o170000 == 0o120000
        archive.extractall(isolated)
    for rel, expected in manifest["members_sha256"].items():
        assert sha(isolated / rel) == expected, rel
    # The source root is tested after runpy, so an editable install cannot silently
    # substitute the original repository for the source extracted from the ZIP.
    probe = (
        "import json,runpy,sys;from pathlib import Path;"
        "sys.dont_write_bytecode=True;"
        "sys.argv=['scripts/v4/demo.py','replay','--output','artifacts/v4/demo/release_replay'];"
        "runpy.run_path('scripts/v4/demo.py',run_name='__main__');"
        "import geopolicy,geopolicy.v4.common;"
        "p=Path(geopolicy.__file__).resolve();root=Path.cwd().resolve();"
        "assert p.is_relative_to(root/'src') and geopolicy.v4.common.ROOT==root;"
        "Path('source_probe.json').write_text(json.dumps({'isolated_source':True,'module_file':str(p),'root':str(root)}));"
    )
    log = OUT / "isolated_replay.log"
    with log.open("w", encoding="utf8") as stream:
        result = subprocess.run(
            [str(ROOT / ".venv-repro/Scripts/python.exe"), "-c", probe],
            cwd=isolated,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    assert result.returncode == 0, log.read_text(encoding="utf8", errors="replace")[
        -5000:
    ]
    checked = read(isolated / "artifacts/v4/demo/release_replay/verification.json")
    assert all(
        checked[k]
        for k in (
            "initial_modalities_exact",
            "actions_exact",
            "physics_exact",
            "summary_exact",
        )
    )
    assert (
        checked["video_sha256"] == manifest["media"]["docs/media/v4_demo.mp4"]["sha256"]
    )
    dump(
        OUT / "replay_verification.json",
        dict(
            **checked,
            source_probe=read(isolated / "source_probe.json"),
            runtime=manifest["runtime"],
            archive_sha256=sha(archive_path),
            no_training_executed=True,
        ),
    )
    print(
        json.dumps(
            dict(replay_exact=True, frames=checked["frames"], isolated_source=True)
        )
    )


def verify():
    snapshot = read(OUT / "preservation.json")
    failures = [
        rel
        for rel, expected in snapshot["protected_files"].items()
        if not (ROOT / rel).exists() or sha(ROOT / rel) != expected
    ]
    assert not failures, failures
    for rel, expected in snapshot["documentation_backups"].items():
        assert sha(OUT / "history" / rel) == expected
    manifest = read(ROOT / "docs/releases/v4_demo.json")
    for name, meta in manifest["assets"].items():
        path = OUT / "assets" / name
        assert sha(path) == meta["sha256"] and path.stat().st_size == meta["size_bytes"]
    for rel, meta in manifest["media"].items():
        assert sha(ROOT / rel) == meta["sha256"]
    for line in (OUT / "assets/SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(OUT / "assets" / name) == digest
    replay = read(OUT / "replay_verification.json")
    assert replay["source_probe"]["isolated_source"]
    cpu = read(OUT / "cpu_tests.json")
    assert cpu["exit_code"] == 0 and not cpu["heavy_modules_loaded"]
    assert cpu["test_file_sha256"] == sha(ROOT / cpu["test_file"])
    dependencies = read(OUT / "pinned_environment_check.json")
    assert dependencies["matched_pins"] == 61 and not dependencies["mismatches"]
    assert dependencies["python_version"] == manifest["runtime"]["python_version"]
    assert dependencies["lock_sha256"] == sha(ROOT / "requirements-lock.txt")
    result = dict(
        verified=False,
        checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        protected_files_checked=len(snapshot["protected_files"]),
        protected_changes=failures,
        results_and_frozen_protocols_unchanged=True,
        original_artifacts_unchanged=True,
        runtime=manifest["runtime"],
        release_assets=manifest["assets"],
        isolated_replay=replay,
        release_manifest_sha256=sha(ROOT / "docs/releases/v4_demo.json"),
        cpu_tests=cpu,
        pinned_environment_check=dependencies,
        publication_status=manifest["status"],
        no_training_executed=True,
        no_cv_edits=True,
    )
    # Materialize the evidence before checking its repository links. Only this
    # new presentation record is written; study outputs stay byte-identical.
    dump(ROOT / "docs/releases/v4_delivery_verification.json", result)
    checks = {}
    for label, checkout in (("local", ROOT), ("repository_copy", OUT / "public_repo")):
        if label == "repository_copy":
            assert not checkout.exists(), "Keep the previous repository-only evidence"
            tracked = (
                subprocess.check_output(
                    [
                        "git",
                        "ls-files",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                        "-z",
                    ],
                    cwd=ROOT,
                )
                .decode("utf8")
                .split("\0")
            )
            for rel in sorted(set(filter(None, tracked))):
                source = (ROOT / rel).resolve()
                assert source.is_relative_to(ROOT) and not source.is_relative_to(
                    ROOT / "artifacts"
                )
                if not source.is_file():
                    continue
                destination = checkout / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            assert not (checkout / "artifacts").exists()
        process = subprocess.run(
            [sys.executable, "scripts/check_publication.py"],
            cwd=checkout,
            capture_output=True,
            text=True,
            encoding="utf8",
        )
        log = OUT / f"publication_{label}.log"
        log.write_text(process.stdout + process.stderr, encoding="utf8")
        assert process.returncode == 0, log.read_text(encoding="utf8")[-5000:]
        payload, _ = json.JSONDecoder().raw_decode(process.stdout)
        checks[label] = payload
    assert checks["local"] == checks["repository_copy"]
    result["publication_checks"] = dict(
        passed=True,
        no_artifacts_in_repository_copy=True,
        **checks,
    )
    result["verified"] = True
    # The copy contains the pre-check record, so refresh only this new file.
    dump(ROOT / "docs/releases/v4_delivery_verification.json", result)
    shutil.copy2(
        ROOT / "docs/releases/v4_delivery_verification.json",
        OUT / "public_repo/docs/releases/v4_delivery_verification.json",
    )
    print(
        json.dumps(
            dict(
                verified=True,
                protected_files_checked=result["protected_files_checked"],
                original_artifacts_unchanged=True,
                publication_status=manifest["status"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("snapshot", "build", "replay", "verify"))
    globals()[parser.parse_args().command]()
