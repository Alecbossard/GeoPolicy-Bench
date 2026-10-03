"""Scientific figures from final summaries and saved physical traces."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    results = json.loads((ROOT / "results/v3/final_summary.json").read_text())
    groups = results["groups"]
    out = ROOT / "docs/v3/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = np.arange(3)
    for prior, offset, color in [
        ("prior", -0.18, "#286fa8"),
        ("no_prior", 0.18, "#87bdd3"),
    ]:
        keys = [f"v3_{view}_{prior}" for view in ("fixed", "wrist", "fusion")]
        values = [100 * groups[key]["strict"] / 150 for key in keys]
        bars = axes[0].bar(
            x + offset,
            values,
            0.34,
            color=color,
            label="Avec prior" if prior == "prior" else "Sans prior",
        )
        for bar, key in zip(bars, keys):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{groups[key]['strict']}/150",
                ha="center",
                fontsize=9,
            )
    axes[0].set(
        xticks=x,
        xticklabels=["Fixe", "Poignet", "Fusion"],
        ylabel="Placements stricts V2 (%)",
        ylim=(0, 112),
        title="Même tâche, données et budget — 3 seeds",
    )
    axes[0].legend(loc="lower right")
    keys = ["v3_original_bc", "v3_fusion_prior", "v3_v1_recipe"]
    labels = [
        "BC original",
        "BC continuation\n+ historique4",
        "Recette diffusion\nV1 réentraînée",
    ]
    for metric, offset, color in [
        ("physical", -0.18, "#568c65"),
        ("strict", 0.18, "#286fa8"),
    ]:
        values = [100 * groups[key][metric] / 150 for key in keys]
        bars = axes[1].bar(
            x + offset,
            values,
            0.34,
            color=color,
            label="Physique V3" if metric == "physical" else "Strict V2",
        )
        for bar, key in zip(bars, keys):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{groups[key][metric]}/150",
                ha="center",
                fontsize=9,
            )
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 112),
        title="Avant/après sur le nouveau test",
    )
    axes[1].legend(loc="lower right")
    fig.suptitle("Un cube, un bac, consigne fixe • 50 scènes nouvelles × 3 seeds")
    fig.savefig(out / "final_comparisons.png", dpi=180)
    fig.savefig(out / "final_comparisons.svg")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex="col", layout="constrained")
    for column, name in enumerate(
        ["final_v3_original_bc_s2", "final_single_fusion_prior_s2"]
    ):
        trace = json.loads(
            (ROOT / f"artifacts/v3/evaluations/{name}/traces/400000.json").read_text()
        )
        t = [r["time_s"] for r in trace]
        axes[0, column].plot(
            t, [r["finger_width_m"] * 1000 for r in trace], color="#286fa8"
        )
        axes[0, column].axhline(45, color="gray", ls="--", lw=1)
        axes[0, column].set(
            title="BC original" if column == 0 else "Continuation + historique4",
            ylabel="Ouverture pince (mm)",
        )
        axes[1, column].plot(t, [r["linear_speed_m_s"] for r in trace], color="#568c65")
        axes[1, column].axhline(0.02, color="gray", ls="--", lw=1)
        axes[1, column].set_yscale("symlog", linthresh=0.02)
        axes[1, column].set(xlabel="Temps simulation (s)", ylabel="Vitesse objet (m/s)")
        axes[0, column].set_ylim(0, 85)
        release_time = next((r["time_s"] for r in trace if r["release_seen"]), None)
        for ax in axes[:, column]:
            if release_time is not None:
                ax.axvline(release_time, color="#8c476f", ls=":", lw=1)
            for r in trace:
                if r["finger_contact"]:
                    ax.axvspan(
                        r["time_s"] - 0.025,
                        r["time_s"] + 0.025,
                        color="#ddaa80",
                        alpha=0.2,
                        lw=0,
                    )
    fig.suptitle(
        "Scène 400000, seed 2 • contacts beige • seuils gris • libération mauve"
    )
    fig.savefig(out / "release_trace.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
