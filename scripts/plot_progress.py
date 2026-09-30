"""Measured preliminary figures; does not imply benchmark completion."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {"font.size": 10, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False}
)
controls = json.loads(Path("artifacts/teacher_controls.json").read_text())
names = ["initial", "scripted_reference", "bc_initial", "ppo_selected"]
values = [np.mean([r["success"] for r in controls[n]]) * 100 for n in names]
fig, ax = plt.subplots(figsize=(6.4, 3.7))
bars = ax.bar(
    ["Untrained", "Scripted", "BC bootstrap", "Selected PPO"],
    values,
    color=["#888888", "#7b61a8", "#dd9b33", "#3288a4"],
)
ax.set_ylim(0, 110)
ax.set_ylabel("Instruction-correct success (%)")
ax.set_title("Teacher controls: 20 held-out validation scenes")
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%", ha="center")
fig.text(
    0.01,
    0.01,
    "PPO is BC initialized with a scripted phase curriculum; no supported gain over BC.",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(out / "teacher_controls.png")
fig.savefig(out / "teacher_controls.svg")
plt.close(fig)
curve = json.loads(Path("artifacts/teacher_main/validation_curve.json").read_text())
fig, ax = plt.subplots(figsize=(6.4, 3.7))
ax.plot(
    [r["transitions"] for r in curve],
    [r["success"] * 100 for r in curve],
    marker="o",
    color="#bc5035",
)
ax.axhline(85, ls="--", color="#3288a4", label="Selected PPO pilot (17/20)")
ax.scatter([34816], [10], color="#bc5035", marker="x", s=60, label="Final continuation (2/20)")
ax.set_ylim(0, 100)
ax.set_xlabel("Total collected PPO transitions")
ax.set_ylabel("Validation success (%)")
ax.set_title("Negative result: longer PPO continuation regresses")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(out / "ppo_negative_continuation.png")
fig.savefig(out / "ppo_negative_continuation.svg")
plt.close(fig)
manifest = json.loads(Path("configs/dataset_manifest.json").read_text())
episodes = [r for r in manifest["episodes"] if r["split"] == "train"]
labels = []
success = []
failure = []
for obj in range(2):
    for goal in range(2):
        rows = [r for r in episodes if r["object_id"] == obj and r["goal_id"] == goal]
        labels.append(["red", "green"][obj] + " → " + ["blue", "yellow"][goal])
        success.append(sum(r["success"] for r in rows))
        failure.append(sum(not r["success"] for r in rows))
fig, ax = plt.subplots(figsize=(6.4, 3.7))
ax.bar(labels, success, label="Success", color="#3288a4")
ax.bar(labels, failure, bottom=success, label="Failure retained", color="#bc5035")
ax.set_ylabel("Independent scene episodes")
ax.set_title("PPO-provenance collection: 250 training scenes")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "dataset_collection.png")
fig.savefig(out / "dataset_collection.svg")
plt.close(fig)
print(out.resolve())
