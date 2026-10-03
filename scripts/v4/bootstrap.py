"""Snapshot history before the explicitly authorized V3 documentation correction."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    target = ROOT / "configs/v4/preservation.json"
    assert not target.exists()
    files = {
        ROOT / p
        for p in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
        if p
    }
    old = json.loads((ROOT / "configs/v3/preservation.json").read_text(encoding="utf8"))
    for p, h in old["files"].items():
        assert digest(ROOT / p) == h, p
        files.add(ROOT / p)
    files.update(
        p
        for p in (ROOT / "artifacts/v3").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    corrections = [
        "docs/v3/README.md",
        "docs/v3/report.md",
        "docs/v3/model_card.md",
        "docs/v3/reproduction.md",
        "docs/v3/PROGRESS.md",
        "scripts/v3/build_report.py",
    ]
    for rel in corrections:
        dst = ROOT / "artifacts/v4/preservation" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        allowed_documentation_corrections=corrections,
        files={p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    replacements = {
        "80 scènes train 10000–10079 et 10 démonstrations de validation 110000–110009.": "80 collectes train (10000–10079), dont **79 démonstrations retenues** :\nla scène 10032, échec de prise, est exclue. 10 démonstrations de validation\n110000–110009 sont conservées séparément.",
        "mêmes 80+10 démonstrations continuées": "mêmes 79 démonstrations train + 10 de validation continuées",
        "mêmes 80 trajectoires": "mêmes 79 trajectoires retenues",
        "Entraînement :80démos": "Entraînement :79 démonstrations retenues sur80collectes",
        "Sur 80 démos / 2 000 updates": "Sur 79 démos retenues sur80collectes / 2 000 updates",
        "mêmes 80+10 démos": "mêmes 79 démos train +10validation",
        "sur les 80 préfixes / 2 000 updates": "sur les 79 préfixes retenus / 2 000 updates",
        "contiennent 80 épisodes train et 10 validation enregistrée.": "contiennent 80 épisodes train collectés, dont79succès réellement utilisés\npour entraînement (10032 exclu), et10de validation enregistrée.\n`--limit80` est un plafond historique et le nom des runs reste inchangé.",
    }
    for rel in corrections:
        path = ROOT / rel
        text = path.read_text(encoding="utf8")
        for a, b in replacements.items():
            text = text.replace(a, b)
        if rel == "docs/v3/README.md":
            text += "\nCorrection du décompte : **79 démonstrations train utilisées sur80collectes**,\nplus10démonstrations de validation. Scène10032 exclue pour échec de prise.\nLes résultats, checkpoints et identités historiques restent inchangés.\n"
        path.write_text(text, encoding="utf8")
    records = json.loads(
        (ROOT / "configs/v3/single_dataset.json").read_text(encoding="utf8")
    )
    result = dict(
        collected_train=80,
        used_train=79,
        excluded=[r for r in records if r["split"] == "train" and not r["success"]],
        validation=10,
        historical_files=len(files),
        corrections=corrections,
    )
    p = ROOT / "results/v4/dataset_correction.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "excluded"}))


if __name__ == "__main__":
    main()
