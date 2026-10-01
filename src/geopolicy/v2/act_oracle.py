"""Validation-only component interventions; never a learned baseline or final score."""

import json
from pathlib import Path
from geopolicy.io import save_json
from .evaluation import evaluate
from .config import read_recipe


def diagnose_components():
    recipe = read_recipe("configs/v2/diagnostic_recipe.json")
    unassisted = json.loads(Path("artifacts/v2/diagnostics/act_v1/rollouts.json").read_text())
    report = dict(
        validation_first_seed=100200,
        episodes=20,
        unassisted_stable_successes=sum(r["stable_success"] for r in unassisted),
        reserved_test_used=False,
        learned_baseline=False,
        interpretation="Ground-truth scripted motion/gripper replaces one component only. A difference measures the effect of that intervention in these scenes, not a unique architectural cause or a learned success gain.",
    )
    for component in ["motion", "gripper"]:
        rows = evaluate(
            recipe,
            "artifacts/main_runs/act_s0/best.pt",
            f"artifacts/v2/act_oracle/{component}",
            100200,
            20,
            oracle_component=component,
        )
        report[component + "_oracle_stable_successes"] = sum(r["stable_success"] for r in rows)
        report[component + "_oracle_success_scene_seeds"] = [
            r["scene_seed"] for r in rows if r["stable_success"]
        ]
    save_json("results/v2/act_component_interventions.json", report)
    return report
