"""Aggregate complete episode-level results, paired uncertainty and figures."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--bootstrap", type=int, default=5000)
a = p.parse_args()
protocol = json.loads(Path("configs/benchmark_protocol.json").read_text())
raw = Path("results/raw")
raw.mkdir(parents=True, exist_ok=True)
fig = Path("results/figures")
fig.mkdir(parents=True, exist_ok=True)
conditions = protocol["evaluation"]["conditions"]
modes = ["mono", "fusion", "act"]
rows = []
cells = {}
summary = []
rng = np.random.default_rng(90230)
for mode in modes:
    for seed in [0, 1, 2]:
        for condition in conditions:
            path = (
                Path("artifacts/final_evaluations")
                / f"{mode}_s{seed}"
                / condition
                / "rollouts.json"
            )
            cell = json.loads(path.read_text())
            expected = (
                protocol["evaluation"]["main_episodes_per_condition"][condition]
                if mode != "act"
                else 20
            )
            assert len(cell) == expected, (path, len(cell), expected)
            assert len({r["scene_seed"] for r in cell}) == expected
            cell = sorted(cell, key=lambda r: r["scene_seed"])
            cells[(mode, seed, condition)] = cell
            rows += cell
            n = len(cell)
            rate = np.mean([r["success"] for r in cell])
            z = 1.959963984540054
            center = (rate + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            summary.append(
                {
                    "mode": mode,
                    "training_seed": seed,
                    "condition": condition,
                    "episodes": n,
                    "success_rate": float(rate),
                    "success_wilson95": [
                        max(0, float(center - half)),
                        min(1, float(center + half)),
                    ],
                    **{
                        key: float(np.mean([r[key] for r in cell]))
                        for key in [
                            "wrong_target",
                            "wrong_destination",
                            "dropped",
                            "collision",
                            "placement_error_m",
                            "placement_error_xyz_m",
                            "wall_seconds",
                            "simulation_seconds",
                            "policy_p50_ms",
                            "policy_p95_ms",
                            "preprocess_p50_ms",
                            "preprocess_p95_ms",
                            "total_p50_ms",
                            "total_p95_ms",
                        ]
                    },
                }
            )
(raw / "student_rollouts.json").write_text(json.dumps(rows, indent=2))
with (raw / "student_rollouts.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
aggregate = []
comparisons = []
for condition in conditions:
    for mode in modes:
        rates = np.array(
            [
                r["success_rate"]
                for r in summary
                if r["mode"] == mode and r["condition"] == condition
            ]
        )
        aggregate.append(
            {
                "mode": mode,
                "condition": condition,
                "mean_success_rate": float(rates.mean()),
                "seed_std": float(rates.std(ddof=1)),
                "seed_rates": rates.tolist(),
            }
        )
    differences = []
    for seed in [0, 1, 2]:
        mono = cells[("mono", seed, condition)]
        fusion = cells[("fusion", seed, condition)]
        assert [r["scene_seed"] for r in mono] == [r["scene_seed"] for r in fusion]
        differences.append(
            np.array([int(f["success"]) - int(m["success"]) for m, f in zip(mono, fusion)])
        )
    boot = []
    # The same physical scene IDs are reused across all three training seeds.
    # Resample shared scene columns together, preserving that crossed design.
    matrix = np.stack(differences)
    for _ in range(a.bootstrap):
        chosen = rng.integers(3, size=3)
        scenes = rng.integers(matrix.shape[1], size=matrix.shape[1])
        boot.append(matrix[np.ix_(chosen, scenes)].mean())
    comparisons.append(
        {
            "condition": condition,
            "fusion_minus_mono_pp": float(100 * np.mean([d.mean() for d in differences])),
            "per_seed_difference_pp": [float(100 * d.mean()) for d in differences],
            "paired_crossed_bootstrap95_pp": (100 * np.percentile(boot, [2.5, 97.5])).tolist(),
            "bootstrap_samples": a.bootstrap,
            "unit": "training seed and shared paired scene columns; never frames",
        }
    )
ood = []
for mode in modes:
    for seed in [0, 1, 2]:
        nominal = {r["scene_seed"]: r for r in cells[(mode, seed, "nominal")]}
        for condition in conditions[1:]:
            pairs = [(nominal[r["scene_seed"]], r) for r in cells[(mode, seed, condition)]]
            ood.append(
                {
                    "mode": mode,
                    "training_seed": seed,
                    "condition": condition,
                    "paired_episodes": len(pairs),
                    "nominal_minus_ood_pp": float(
                        100 * np.mean([int(n["success"]) - int(o["success"]) for n, o in pairs])
                    ),
                }
            )
report = {
    "per_seed": summary,
    "aggregate": aggregate,
    "paired_fusion_comparison": comparisons,
    "ood_drop": ood,
    "uncertainty_note": "3 training seeds give limited estimation of training variability. Paired crossed bootstrap resamples training seeds and shared scene columns; it is descriptive. Test scene family is narrow; secondary20-scene cells exploratory.",
    "protocol": protocol,
}
Path("results/benchmark_summary.json").write_text(json.dumps(report, indent=2))
with Path("results/per_seed.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
labels = ["Nominal", "Occlusion", "Depth noise/holes", "Fixed view missing", "Extrinsic error"]
colors = {"mono": "#4567ae", "fusion": "#e18432", "act": "#48a28a"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
f, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(conditions))
width = 0.24
for i, mode in enumerate(modes):
    entries = [
        next(r for r in aggregate if r["mode"] == mode and r["condition"] == c) for c in conditions
    ]
    ax.bar(
        x + (i - 1) * width,
        [100 * r["mean_success_rate"] for r in entries],
        width,
        color=colors[mode],
        label=mode,
        yerr=[100 * r["seed_std"] for r in entries],
        capsize=3,
    )
ax.set_xticks(x, labels)
ax.set_ylim(0, 105)
ax.set_ylabel("Instruction-correct success (%)")
ax.legend()
ax.set_title("Three training seeds; error bars = seed standard deviation")
f.text(
    0.08,
    0.01,
    "Mono/fusion: n=100 per seed nominal/occlusion/missing, n=20 depth/extrinsics. ACT: n=20 throughout.",
    fontsize=8,
)
f.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ["png", "svg"]:
    f.savefig(fig / f"success_rates.{ext}", dpi=180)
plt.close(f)
f, ax = plt.subplots(figsize=(9, 4))
values = np.array([r["fusion_minus_mono_pp"] for r in comparisons])
ci = np.array([r["paired_crossed_bootstrap95_pp"] for r in comparisons])
ax.errorbar(
    np.arange(5),
    values,
    yerr=np.stack([values - ci[:, 0], ci[:, 1] - values]),
    fmt="o",
    capsize=4,
    color=colors["fusion"],
)
ax.axhline(0, color="gray", linestyle="--")
ax.set_xticks(np.arange(5), labels)
ax.set_ylabel("Fusion − mono success (pp)")
ax.set_title("Paired scene differences; crossed bootstrap 95% intervals")
f.tight_layout()
for ext in ["png", "svg"]:
    f.savefig(fig / f"paired_fusion_difference.{ext}", dpi=180)
plt.close(f)
f, axes = plt.subplots(1, 3, figsize=(12, 3.7))
training = []
for ax, mode in zip(axes, modes):
    for seed in [0, 1, 2]:
        root = Path("artifacts/main_runs") / f"{mode}_s{seed}"
        curve = json.loads((root / "learning_curve.json").read_text())
        val = [r for r in curve if "validation_loss" in r]
        ax.plot(
            [r["update"] for r in val], [r["validation_loss"] for r in val], label=f"seed {seed}"
        )
        manifest = json.loads((root / "manifest.json").read_text())
        training.append(dict(run=f"{mode}_s{seed}", **manifest))
    ax.set_title(mode)
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("EMA offline validation loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
f.suptitle("Losses have different definitions across architectures; compare behavior separately")
f.tight_layout()
for ext in ["png", "svg"]:
    f.savefig(fig / f"student_learning_curves.{ext}", dpi=180)
plt.close(f)
Path("results/training_manifests.json").write_text(json.dumps(training, indent=2))
print(
    json.dumps(
        {"rollouts": len(rows), "aggregate": aggregate, "paired_comparison": comparisons}, indent=2
    )
)
