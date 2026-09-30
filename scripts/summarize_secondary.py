"""Counterfactuals, controlled single-seed ablations and exploratory VLA."""

import json
import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def raw_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


report = {}
counter = []
all_counter_rows = []
for mode in ["mono", "fusion"]:
    for seed in [0, 1, 2]:
        rows = []
        for obj in [0, 1]:
            for goal in [0, 1]:
                path = (
                    Path("artifacts/counterfactual_evaluations")
                    / f"{mode}_s{seed}"
                    / f"object{obj}_goal{goal}"
                    / "rollouts.json"
                )
                cell = json.loads(path.read_text())
                assert len(cell) == 10
                rows += cell
        by_scene = {
            scene: [r for r in rows if r["scene_seed"] == scene] for scene in range(200100, 200110)
        }
        all_counter_rows.extend(rows)
        all_four = []
        object_effect = []
        goal_effect = []
        for scene, items in by_scene.items():
            assert len(items) == 4
            assert (
                len({r["initial_rgbd_robot_camera_pose_sha256"] for r in items}) == 1
            ), "Instruction changed initial sensors"
            all_four.append(all(r["success"] for r in items))
            lookup = {(r["object_id"], r["goal_id"]): r for r in items}
            for fixed in [0, 1]:
                object_effect.append(
                    np.linalg.norm(
                        np.array(lookup[0, fixed]["first_action"])
                        - lookup[1, fixed]["first_action"]
                    )
                )
                goal_effect.append(
                    np.linalg.norm(
                        np.array(lookup[fixed, 0]["first_action"])
                        - lookup[fixed, 1]["first_action"]
                    )
                )
        counter.append(
            {
                "mode": mode,
                "training_seed": seed,
                "scenes": 10,
                "rollouts": 40,
                "success_count": sum(r["success"] for r in rows),
                "scenes_all_four_success": sum(all_four),
                "initial_observations_identical_across_instructions": True,
                "mean_first_action_l2_change_object_instruction": float(np.mean(object_effect)),
                "mean_first_action_l2_change_goal_instruction": float(np.mean(goal_effect)),
                "note": "First-action differences demonstrate conditioning, not correct completion; all-four success is the stricter behavioral measure.",
            }
        )
        out = Path("results/raw") / f"counterfactual_{mode}_s{seed}.json"
        out.write_text(json.dumps(rows, indent=2))
report["counterfactuals"] = counter
assert len(all_counter_rows) == 240
for scene in range(200100, 200110):
    initial = [r for r in all_counter_rows if r["scene_seed"] == scene]
    assert len(initial) == 24
    assert len({r["initial_rgbd_robot_camera_pose_sha256"] for r in initial}) == 1
assert len({r["initial_rgbd_robot_camera_pose_sha256"] for r in all_counter_rows}) == 10
raw_csv("results/raw/counterfactual_rollouts.csv", all_counter_rows)
report["counterfactual_initial_observation_audit"] = {
    "records": 240,
    "distinct_scenes": 10,
    "identical_across_all_four_instructions_both_modes_and_three_training_seeds": True,
}
baseline = json.loads(
    Path("artifacts/final_evaluations/fusion_s0/nominal/rollouts.json").read_text()
)[:20]
baseline_missing = json.loads(
    Path("artifacts/final_evaluations/fusion_s0/fixed_camera_missing/rollouts.json").read_text()
)[:20]
ablations = []
all_ablation_rows = []
for name in ["data50", "view_dropout", "no_voxel", "onnx"]:
    rows = json.loads(Path(f"artifacts/secondary_evaluations/{name}/rollouts.json").read_text())
    Path(f"results/raw/secondary_{name}.json").write_text(json.dumps(rows, indent=2))
    all_ablation_rows.extend(dict(row, variant=name) for row in rows)
    for condition in sorted({r["condition"] for r in rows}):
        cell = sorted(
            [r for r in rows if r["condition"] == condition], key=lambda r: r["scene_seed"]
        )
        reference = baseline_missing if condition == "fixed_camera_missing" else baseline
        assert len(cell) == 20 and [r["scene_seed"] for r in cell] == [
            r["scene_seed"] for r in reference
        ]
        ablations.append(
            {
                "variant": name,
                "condition": condition,
                "training_seed": 0,
                "episodes": 20,
                "success_count": sum(r["success"] for r in cell),
                "baseline200_success_count": sum(r["success"] for r in reference),
                "paired_success_difference_pp": float(
                    100
                    * np.mean(
                        [int(r["success"]) - int(b["success"]) for r, b in zip(cell, reference)]
                    )
                ),
                "success_labels_matching_baseline": sum(
                    r["success"] == b["success"] for r, b in zip(cell, reference)
                ),
                "collision_labels_matching_baseline": sum(
                    r["collision"] == b["collision"] for r, b in zip(cell, reference)
                ),
                "max_placement_error_difference_m": max(
                    abs(r["placement_error_m"] - b["placement_error_m"])
                    for r, b in zip(cell, reference)
                ),
            }
        )
report["single_seed_ablations"] = ablations
assert len(all_ablation_rows) == 100
raw_csv("results/raw/secondary_rollouts.csv", all_ablation_rows)
report["ablation_scope"] = {
    "data50": "First50 successful subset vs200, identical8000-update budget; one seed, not a data-efficiency curve.",
    "view_dropout": "20% probability of one randomly selected retained view; separate checkpointed dropout RNG preserves sampled frame sequence.",
    "no_voxel": "Inference-only voxel removal, unchanged crop/512point budget/model; not a retrained filtering study.",
    "onnx": "Only denoiser exported; same checkpoint/point encoder/DDIM. Physics need not be bit-exact.",
}
vla = json.loads(Path("artifacts/smolvla_test/rollouts.json").read_text())
assert len(vla) == 20
assert {r["scene_seed"] for r in vla} == set(range(200000, 200020))
assert all(r["condition"] == "nominal" for r in vla)
Path("results/raw/smolvla_nominal.json").write_text(json.dumps(vla, indent=2))
raw_csv("results/raw/smolvla_nominal.csv", vla)
report["smolvla"] = {
    "training_seed": 0,
    "nominal_episodes": 20,
    "success_count": sum(r["success"] for r in vla),
    "wrong_target_count": sum(r["wrong_target"] for r in vla),
    "wrong_destination_count": sum(r["wrong_destination"] for r in vla),
    "drop_count": sum(r["dropped"] for r in vla),
    "collision_count": sum(r["collision"] for r in vla),
    "mean_terminal_xy_error_m": float(np.mean([r["placement_error_m"] for r in vla])),
    "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in vla])),
    "mean_simulation_seconds": float(np.mean([r["simulation_seconds"] for r in vla])),
    "mean_episode_preprocess_p50_ms": float(np.mean([r["preprocess_p50_ms"] for r in vla])),
    "mean_episode_preprocess_p95_ms": float(np.mean([r["preprocess_p95_ms"] for r in vla])),
    "mean_episode_policy_p50_ms": float(np.mean([r["policy_p50_ms"] for r in vla])),
    "mean_episode_policy_p95_ms": float(np.mean([r["policy_p95_ms"] for r in vla])),
    "mean_episode_total_p50_ms": float(np.mean([r["total_p50_ms"] for r in vla])),
    "mean_episode_total_p95_ms": float(np.mean([r["total_p95_ms"] for r in vla])),
    "not_budget_matched": "500 optimizer steps/accum4/batch1, pretrained450M vs compact8000/batch32/no pretraining. Exploratory one seed.",
    "ood_deferred_if_zero_nominal": not any(r["success"] for r in vla),
}
ood_path = Path("artifacts/smolvla_test_ood/rollouts.json")
if ood_path.exists():
    ood = json.loads(ood_path.read_text())
    assert len(ood) == 80
    Path("results/raw/smolvla_ood.json").write_text(json.dumps(ood, indent=2))
    report["smolvla"]["ood"] = {
        c: sum(r["success"] for r in ood if r["condition"] == c)
        for c in sorted({r["condition"] for r in ood})
    }
Path("results/secondary_summary.json").write_text(json.dumps(report, indent=2))
fig = Path("results/figures")
fig.mkdir(parents=True, exist_ok=True)
f, ax = plt.subplots(figsize=(9, 4))
labels = [r["variant"] + "\n" + r["condition"] for r in ablations]
ax.bar(np.arange(len(labels)), [100 * r["success_count"] / 20 for r in ablations], color="#e18432")
for i, r in enumerate(ablations):
    ax.plot(
        [i - 0.35, i + 0.35],
        [100 * r["baseline200_success_count"] / 20] * 2,
        color="#4567ae",
        linewidth=3,
    )
ax.set_xticks(np.arange(len(labels)), labels)
ax.set_ylim(0, 105)
ax.set_ylabel("Success (%)")
ax.set_title("Exploratory seed 0 / 20 paired scenes; blue line = 200-demo baseline")
f.tight_layout()
for ext in ["png", "svg"]:
    f.savefig(fig / f"secondary_ablations.{ext}", dpi=180)
plt.close(f)
print(json.dumps(report, indent=2))
