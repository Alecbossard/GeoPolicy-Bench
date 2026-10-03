"""Branch the completed pilot weights into separate resumable main run paths."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    for group in ("fixed_clean", "fixed_aug", "fusion_clean", "fusion_aug"):
        name = group + "_s0"
        src = ROOT / "artifacts/v4/runs" / name
        dst = ROOT / "artifacts/v4/main_runs" / name
        assert src.is_relative_to(ROOT / "artifacts/v4") and dst.is_relative_to(
            ROOT / "artifacts/v4"
        )
        assert not dst.exists()
        assert (
            json.loads((src / "manifest.json").read_text())["completed_update"] == 1000
        )
        shutil.copytree(src, dst)
        old = ROOT / "results/v4/training" / f"{name}.json"
        archived = ROOT / "results/v4/pilot_training" / old.name
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, archived)
    jobs = [
        ["train", "--scope", "main", "--view", v, "--seed", "0"]
        + (["--augmented"] if a else [])
        for v in ("fixed", "fusion")
        for a in (False, True)
    ]
    (ROOT / "configs/v4/seed0_complete_jobs.json").write_text(
        json.dumps(jobs, indent=2), encoding="utf8"
    )
    jobs = [
        [
            "eval",
            "--name",
            f"pilot2000_{v}_{a}_{c}",
            "--checkpoint",
            f"artifacts/v4/main_runs/{v}_{a}_s0/best.pt",
            "--condition",
            c,
            "--first",
            "210000",
            "--episodes",
            "10",
        ]
        for v in ("fixed", "fusion")
        for a in ("clean", "aug")
        for c in json.loads((ROOT / "configs/v4/plan.json").read_text())[
            "pilot_conditions"
        ]
    ]
    (ROOT / "configs/v4/pilot2000_eval_jobs.json").write_text(
        json.dumps(jobs, indent=2), encoding="utf8"
    )
    print("Pilot checkpoints preserved; four separate main continuations prepared")


if __name__ == "__main__":
    main()
