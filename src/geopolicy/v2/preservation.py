"""Check original tracked metrics, source files and selected checkpoint identities."""

import json
import subprocess
from pathlib import Path
from .config import file_hash
from geopolicy.io import save_json


def preserve_v1():
    path = Path("configs/v2/v1_inventory.json")
    if path.exists():
        inventory = json.loads(path.read_text())
    else:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "geopolicy-v1-2026-09-30"], text=True
        ).splitlines()
        names = [
            n for n in names if n.startswith(("results/", "configs/", "src/", "scripts/", "tests/"))
        ]
        inventory = {
            "tag": "geopolicy-v1-2026-09-30",
            "commit": "542f132",
            "original_files_sha256": {n: file_hash(n) for n in names},
            "checkpoint_sha256": {
                r["checkpoint"]: r["sha256"]
                for r in json.loads(Path("results/checkpoint_audit.json").read_text())["main_runs"]
            },
        }
        inventory["checkpoint_sha256"]["artifacts/smolvla_s0/best.pt"] = json.loads(
            Path("artifacts/smolvla_test/evaluation_identity.json").read_text()
        )["checkpoint_sha256"]
        inventory["checkpoint_sha256"][
            "artifacts/teacher_selected.zip"
        ] = "384576a980ee55a6969d1fe00caf85cf0323dccbcf5f1fcc8562ea4d9e607d63"
        save_json(path, inventory)
    if "all_original_checkpoint_sha256" not in inventory:
        inventory["all_original_checkpoint_sha256"] = {
            p.as_posix(): file_hash(p)
            for extension in ["*.pt", "*.zip"]
            for p in Path("artifacts").rglob(extension)
            if "v2" not in p.parts
        }
        save_json(path, inventory)
    failures = [n for n, h in inventory["original_files_sha256"].items() if file_hash(n) != h]
    failures += [n for n, h in inventory["checkpoint_sha256"].items() if file_hash(n) != h]
    failures += [
        n for n, h in inventory["all_original_checkpoint_sha256"].items() if file_hash(n) != h
    ]
    assert not failures, f"V1 files changed: {failures}"
    report = {
        "version1_preserved": True,
        "tag": inventory["tag"],
        "original_files": len(inventory["original_files_sha256"]),
        "selected_checkpoints": len(inventory["checkpoint_sha256"]),
        "all_original_checkpoints": len(inventory["all_original_checkpoint_sha256"]),
    }
    save_json("results/v2/v1_preservation.json", report)
    return report
