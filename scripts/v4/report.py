"""Build V4 results and documentation from complete executed test rows only.

No Torch, GPU, simulator, model selection, or historical artifact mutation.
The benchmark conditions are fixed; uncertainty resamples training seeds and
scenes, preserving every pairing and the shared scene across all conditions.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("fixed_clean", "fixed_aug", "fusion_clean", "fusion_aug")
CONDITIONS = (
    "nominal",
    "occlusion25",
    "occlusion60",
    "absent",
    "depth003",
    "depth010",
    "depth025",
    "missing30",
    "missing70",
    "missing90",
)
METRICS = ("physical_success", "strict_v2_success")
SEEDS = (0, 1, 2)
SCENES = tuple(range(500000, 500020))
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 81
LABELS = {
    "fixed_clean": "Fixe, sans augmentation",
    "fixed_aug": "Fixe, avec augmentations",
    "fusion_clean": "Fusion, sans augmentation",
    "fusion_aug": "Fusion, avec augmentations",
    "nominal": "Nominal",
    "occlusion25": "Occultation fixe 25 %",
    "occlusion60": "Occultation fixe 60 %",
    "absent": "Caméra fixe absente",
    "depth003": "Bruit profondeur σ=3 mm",
    "depth010": "Bruit profondeur σ=10 mm",
    "depth025": "Bruit profondeur σ=25 mm",
    "missing30": "Points manquants 30 %",
    "missing70": "Points manquants 70 %",
    "missing90": "Points manquants 90 %",
    "mean_perturbed": "Moyenne des neuf conditions altérées",
}
CONTRASTS = (
    ("fixed_aug", "fixed_clean"),
    ("fusion_aug", "fusion_clean"),
    ("fusion_clean", "fixed_clean"),
    ("fusion_aug", "fixed_aug"),
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write(path: Path, value: Any) -> None:
    """Atomic output, confined to the report's V4 namespaces."""
    resolved = path.resolve()
    allowed = [(ROOT / d).resolve() for d in ("results/v4", "docs/v4")]
    if not any(resolved.is_relative_to(d) for d in allowed):
        raise ValueError(f"Output outside V4 report namespace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, str):
        temporary.write_text(value, encoding="utf8")
    else:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf8",
        )
    os.replace(temporary, path)


def bootstrap_indices() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = rng.integers(len(SEEDS), size=(BOOTSTRAP_DRAWS, len(SEEDS)))
    scenes = rng.integers(len(SCENES), size=(BOOTSTRAP_DRAWS, len(SCENES)))
    return seeds, scenes


def interval(matrix: np.ndarray, indices: tuple[np.ndarray, np.ndarray]) -> dict:
    if matrix.shape != (len(SEEDS), len(SCENES)) or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap requires a finite, complete seed/scene matrix")
    seeds, scenes = indices
    values = matrix[seeds[:, :, None], scenes[:, None, :]].mean((1, 2)) * 100
    return {
        "estimate_percent_or_pp": float(matrix.mean() * 100),
        "descriptive_95_percent_or_pp": np.percentile(values, [2.5, 97.5]).tolist(),
        "training_seeds": len(SEEDS),
        "unique_scenes": len(SCENES),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def load_rows(protocol: dict) -> tuple[list[dict], dict[str, str]]:
    indexed = {}
    inputs = {}
    registry = {r["checkpoint"]: r for r in protocol["registry"]}
    for path in sorted((ROOT / "results/v4/test").glob("*.json")):
        result = read(path)
        if (
            not isinstance(result, dict)
            or "identity" not in result
            or "rollouts" not in result
        ):
            raise ValueError(f"Unexpected test file schema: {relative(path)}")
        identity = result["identity"]
        checkpoint = identity.get("checkpoint")
        if (
            identity.get("split") != "test"
            or identity.get("first") != SCENES[0]
            or identity.get("episodes") != len(SCENES)
            or checkpoint not in registry
            or identity.get("checkpoint_sha256") != registry[checkpoint]["sha256"]
        ):
            raise ValueError(
                f"Test identity differs from frozen registry: {relative(path)}"
            )
        inputs[relative(path)] = sha(path)
        for row in result["rollouts"]:
            key = (
                row["group"],
                row["condition"],
                row["training_seed"],
                row["scene_seed"],
            )
            if key in indexed:
                raise ValueError(f"Duplicate final rollout: {key}")
            if row.get("diagnostic_oracle", False):
                raise ValueError(f"Student test contains an oracle row: {key}")
            record = registry[checkpoint]
            if (
                row["group"] != record["group"]
                or row["training_seed"] != record["training_seed"]
            ):
                raise ValueError(
                    f"Rollout labels differ from checkpoint registry: {key}"
                )
            if (
                row.get("condition") != identity["condition"]
                or row.get("checkpoint_sha256") != identity["checkpoint_sha256"]
            ):
                raise ValueError(f"Rollout identity differs from its file: {key}")
            for metric in METRICS:
                if type(row[metric]) is not bool:
                    raise ValueError(f"Nonboolean outcome {metric}: {key}")
            for field in ("initial_clean_hash", "initial_corrupted_hash"):
                digest = row.get(field)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError(f"Invalid initial modality digest {field}: {key}")
            indexed[key] = row
    expected = {
        (group, condition, seed, scene)
        for group in GROUPS
        for condition in CONDITIONS
        for seed in SEEDS
        for scene in SCENES
    }
    absent, unexpected = expected - indexed.keys(), indexed.keys() - expected
    if absent or unexpected:
        diagnostic = {
            "complete": False,
            "expected_rollouts": len(expected),
            "observed_rollouts": len(indexed),
            "missing_cells": sorted(absent),
            "unexpected_cells": sorted(unexpected),
        }
        write(ROOT / "results/v4/report_completeness.json", diagnostic)
        raise ValueError(
            f"Final matrix incomplete: {len(absent)} missing, {len(unexpected)} unexpected; "
            "results/v4/report_completeness.json saved. No report fabricated."
        )
    rows = [indexed[key] for key in sorted(expected)]
    return rows, inputs


def training_evidence(plan: dict) -> dict:
    expected_data, reference_config = None, None
    inputs, cells = {}, {}
    for group in GROUPS:
        for seed in SEEDS:
            key = f"{group}_s{seed}"
            path = ROOT / "results/v4/training" / f"{key}.json"
            result = read(path)
            manifest = result["manifest"]
            cfg = manifest["config"]
            view, augmentation = group.split("_")
            if (
                manifest["completed_update"] != plan["training"]["updates"]
                or manifest["actual_train_episodes"] != 79
                or manifest["training_frames"] != 9986
                or manifest["validation_episodes"] != 10
                or cfg["view"] != view
                or cfg["augmented"] != (augmentation == "aug")
                or cfg["seed"] != seed
            ):
                raise ValueError(
                    f"Final training manifest incomplete or different: {key}"
                )
            data = manifest["training_data"]
            comparable = {
                k: v for k, v in cfg.items() if k not in ("view", "augmented", "seed")
            }
            if expected_data is None:
                expected_data, reference_config = data, comparable
            if data != expected_data or comparable != reference_config:
                raise ValueError(
                    f"Unmatched final demonstrations or training budget/config: {key}"
                )
            inputs[relative(path)] = sha(path)
            cells[key] = {
                "actual_train_episodes": manifest["actual_train_episodes"],
                "training_frames": manifest["training_frames"],
                "completed_update": manifest["completed_update"],
                "best_checkpoint_sha256": manifest["best_checkpoint_sha256"],
            }
    return {"cells": cells, "input_sha256": inputs, "matched_data_and_budget": True}


def verify_identity(rows: list[dict]) -> dict:
    """Require exact policy inputs; retain and explain every raw-modality difference."""
    indexed = {
        (r["group"], r["condition"], r["training_seed"], r["scene_seed"]): r
        for r in rows
    }
    mismatches = []
    for row in rows:
        scene, condition = row["scene_seed"], row["condition"]
        clean_ref = indexed[("fixed_clean", "nominal", 0, scene)]
        corrupted_ref = indexed[("fixed_clean", condition, 0, scene)]
        for field, reference in (
            ("initial_clean_hash", clean_ref),
            ("initial_corrupted_hash", corrupted_ref),
        ):
            if row[field] != reference[field]:
                mismatches.append(
                    {
                        "group": row["group"],
                        "condition": condition,
                        "training_seed": row["training_seed"],
                        "scene_seed": scene,
                        "field": field,
                        "observed": row[field],
                        "reference": reference[field],
                    }
                )
    qc_path = ROOT / "results/v4/initial_sensor_qc.json"
    qc = read(qc_path)
    assert qc["checked_rollouts"] == len(rows) == 2400
    for rel, expected in qc["raw_result_sha256"].items():
        assert sha(ROOT / rel) == expected, f"QC raw input changed: {rel}"
    assert len(qc["raw_result_sha256"]) == 120
    assert qc["all_effective_policy_initial_pairs_exact"]
    assert not qc["effective_mismatches"]
    # RGB images are retained diagnostics. live_batch consumes sampled XYZRGB/masks
    # and robot state; those arrays must match exactly, with no numeric tolerance.
    result = {
        "all_initial_pairs_exact": not mismatches,
        "all_raw_initial_modalities_exact": qc["all_raw_initial_modalities_exact"],
        "all_effective_policy_initial_pairs_exact": qc[
            "all_effective_policy_initial_pairs_exact"
        ],
        "quality_control_sha256": sha(qc_path),
        "raw_modality_mismatches": qc["raw_modality_mismatches"],
        "rollouts_checked": len(rows),
        "unique_clean_scenes": len(SCENES),
        "unique_corrupted_scene_conditions": len(SCENES) * len(CONDITIONS),
        "reference": "fixed_clean / seed0; nominal for clean packets",
        "mismatches": mismatches,
        "treatment": (
            "All 2400 raw outcomes retained, with no exclusion or rerun. "
            "Post-test QC requires bit-exact clean/corrupted sampled points, masks, "
            "state, depth and calibration. One raw RGB component differs by one "
            "8-bit level outside the effective sampled-point representation. "
            "The raw-image exception remains false under the strict raw hash check; "
            "its cause is unproven. The policy never receives the full RGB image."
        ),
    }
    write(ROOT / "results/v4/report_sensor_identity.json", result)
    if any(m["field"] == "initial_corrupted_hash" for m in mismatches):
        raise ValueError(
            f"{len(mismatches)} initial packet mismatches; "
            "effective corruptions do not match; paired report refused."
        )
    return result


def summarize(rows: list[dict], inputs: dict[str, str], identity: dict) -> dict:
    matrices = {}
    summaries = {}
    indices = bootstrap_indices()
    by_cell = collections.defaultdict(list)
    for row in rows:
        by_cell[(row["group"], row["condition"])].append(row)
    for group in GROUPS:
        matrices[group] = {}
        summaries[group] = {}
        for condition in CONDITIONS:
            selected = by_cell[(group, condition)]
            selected.sort(key=lambda r: (r["training_seed"], r["scene_seed"]))
            matrices[group][condition] = {}
            cell = {"rollouts": len(selected)}
            for metric in METRICS:
                matrix = np.asarray(
                    [r[metric] for r in selected], dtype=np.float64
                ).reshape(len(SEEDS), len(SCENES))
                matrices[group][condition][metric] = matrix
                cell[metric] = {
                    "successes": int(matrix.sum()),
                    "per_seed_successes": matrix.sum(1).astype(int).tolist(),
                    **interval(matrix, indices),
                }
            cell["physical_failure_stages"] = dict(
                collections.Counter(
                    str(r["failure_stage"])
                    for r in selected
                    if not r["physical_success"]
                )
            )
            cell["posture_only_failures"] = sum(
                r["physical_success"] and not r["strict_v2_success"] for r in selected
            )
            summaries[group][condition] = cell
        matrices[group]["mean_perturbed"] = {
            metric: np.stack(
                [matrices[group][condition][metric] for condition in CONDITIONS[1:]]
            ).mean(0)
            for metric in METRICS
        }
        summaries[group]["mean_perturbed"] = {
            "conditions": list(CONDITIONS[1:]),
            "rollouts": len(SEEDS) * len(SCENES) * (len(CONDITIONS) - 1),
            "interpretation": "Equal weight for nine fixed perturbation settings; not independent scenes",
            **{
                metric: {
                    "successes": sum(
                        summaries[group][condition][metric]["successes"]
                        for condition in CONDITIONS[1:]
                    ),
                    "per_seed_rate_percent": (
                        matrices[group]["mean_perturbed"][metric].mean(1) * 100
                    ).tolist(),
                    **interval(matrices[group]["mean_perturbed"][metric], indices),
                }
                for metric in METRICS
            },
        }
    contrasts = {}
    for first, second in CONTRASTS:
        key = first + "_minus_" + second
        contrasts[key] = {}
        for condition in (*CONDITIONS, "mean_perturbed"):
            contrasts[key][condition] = {
                metric: interval(
                    matrices[first][condition][metric]
                    - matrices[second][condition][metric],
                    indices,
                )
                for metric in METRICS
            }
    return {
        "task": "single_cube_single_tray_fixed_instruction",
        "data": {
            "collected_train": 80,
            "used_train": 79,
            "excluded_scene_seed": 10032,
            "exclusion_reason": "failed scripted grasp",
            "recorded_validation": 10,
            "continued_training_frames": 9986,
        },
        "test": {
            "first_scene": SCENES[0],
            "last_scene": SCENES[-1],
            "unique_scenes": len(SCENES),
            "training_seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "rollouts": len(rows),
        },
        "groups": summaries,
        "paired_contrasts": contrasts,
        "sensor_identity": identity,
        "input_sha256": inputs,
        "statistics": {
            "method": "Descriptive crossed training-seed/scene paired bootstrap",
            "draws": BOOTSTRAP_DRAWS,
            "rng_seed": BOOTSTRAP_SEED,
            "level": 0.95,
            "multiplicity_correction": False,
            "condition_resampling": False,
            "uncertainty": (
                "Three training seeds and twenty repeated scenes; conditions are fixed. "
                "Ceiling/floor bootstrap intervals cannot reveal unobserved outcomes. "
                "No equivalence or noninferiority claim from a nonsignificant contrast."
            ),
        },
        "scope": (
            "Synthetic degradation after sensor acquisition, with an ideal simulated camera "
            "calibration and known object colors. No real sensor or robot transfer evaluated."
        ),
    }


def figures(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ROOT / "docs/v4/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    families = (
        (
            "Caméra fixe occultée / absente",
            [0, 0.25, 0.60, 1],
            ["nominal", "occlusion25", "occlusion60", "absent"],
            "Fraction masquée (1 = caméra absente)",
        ),
        (
            "Bruit de profondeur",
            [0, 3, 10, 25],
            ["nominal", "depth003", "depth010", "depth025"],
            "Écart-type axial (mm)",
        ),
        (
            "Points manquants",
            [0, 0.3, 0.7, 0.9],
            ["nominal", "missing30", "missing70", "missing90"],
            "Probabilité de suppression",
        ),
    )
    styles = {
        "fixed_clean": ("#b34d47", "--", "o"),
        "fixed_aug": ("#b34d47", "-", "s"),
        "fusion_clean": ("#216b98", "--", "o"),
        "fusion_aug": ("#216b98", "-", "s"),
    }
    for metric, stem, ylabel in (
        ("physical_success", "robustness_physical", "Placement physique stable (%)"),
        ("strict_v2_success", "robustness_strict", "Placement strict V2 (%)"),
    ):
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 4), layout="constrained", sharey=True
        )
        for ax, (title, x, conditions, xlabel) in zip(axes, families):
            for group in GROUPS:
                cells = [summary["groups"][group][c][metric] for c in conditions]
                values = [cell["estimate_percent_or_pp"] for cell in cells]
                bounds = np.asarray(
                    [cell["descriptive_95_percent_or_pp"] for cell in cells]
                )
                color, linestyle, marker = styles[group]
                ax.plot(
                    x,
                    values,
                    color=color,
                    ls=linestyle,
                    marker=marker,
                    label=LABELS[group],
                )
                ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=0.07)
            ax.set(title=title, xlabel=xlabel, ylim=(-2, 102), xticks=x)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8, loc="lower left")
        fig.suptitle(
            "20 scènes réservées × 3 seeds ; bandes : bootstrap croisé descriptif à 95 %"
        )
        fig.savefig(out / f"{stem}.png", dpi=180)
        fig.savefig(out / f"{stem}.svg")
        plt.close(fig)


def metric_text(cell: dict, metric: str = "physical_success") -> str:
    data = cell[metric]
    denominator = cell["rollouts"]
    return f"{data['successes']}/{denominator} ({data['estimate_percent_or_pp']:.1f} %)"


def difference_text(result: dict) -> str:
    lo, hi = result["descriptive_95_percent_or_pp"]
    return f"{result['estimate_percent_or_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def available_link(label: str, target: Path) -> str | None:
    if not target.exists():
        return None
    return f"[{label}](../../{relative(target)})"


def video_index(rows: list[dict]) -> None:
    """List the eight predeclared captures, preserving successes and failures."""
    selected = {
        (row["group"], row["condition"]): row
        for row in rows
        if row["training_seed"] == 0
        and row["scene_seed"] == 500000
        and row["condition"] in ("nominal", "absent")
    }
    expected = {
        (group, condition) for group in GROUPS for condition in ("nominal", "absent")
    }
    if set(selected) != expected:
        raise ValueError("Missing outcomes for the eight predeclared video captures")
    entries = []
    for group in GROUPS:
        for condition in ("nominal", "absent"):
            name = f"final_{group}_s0_{condition}"
            capture = ROOT / f"artifacts/v4/evaluations/{name}/scene_500000.mp4"
            if not capture.is_file() or capture.stat().st_size == 0:
                raise ValueError(
                    f"Missing or empty predeclared capture: {relative(capture)}"
                )
            raw = ROOT / f"results/v4/test/{name}.json"
            row = selected[(group, condition)]
            physical = "réussi" if row["physical_success"] else "échoué"
            strict = "réussi" if row["strict_v2_success"] else "échoué"
            entries.append(
                f"- **{LABELS[group]} — {LABELS[condition]} :** "
                f"[vidéo](../../{relative(capture)}), physique **{physical}**, "
                f"strict V2 **{strict}**. [Résultat brut](../../{relative(raw)}). "
                f"Chemin : `{relative(capture)}`. SHA256 : `{sha(capture)}`."
            )
    demo_paths = sorted((ROOT / "artifacts/v4/demo").rglob("*.mp4"))
    demo = (
        available_link("démo principale", demo_paths[0])
        if demo_paths
        else "démo principale décrite dans le [guide de reproduction](reproduction.md)"
    )
    content = "\n".join(
        [
            "# V4 — index des huit captures préspécifiées",
            "",
            "Les huit clips sont fixés par le plan des jobs avant l'accès au test :",
            "seed 0, première scène réservée 500000, conditions nominale et caméra fixe",
            "absente pour chacun des quatre groupes. Aucun clip n'est choisi en fonction",
            "de sa réussite ; les échecs restent présentés. Une seule scène illustrée",
            "par clip ne constitue pas une estimation de performance.",
            "",
            "Ces captures montrent le **RGB physique brut du simulateur**, avec les deux",
            "caméras affichées. La corruption est appliquée à la représentation de points",
            "après acquisition, crop, voxelisation et échantillonnage. L'image RGB affichée",
            "reste donc disponible à l'écran lorsque les points de la caméra fixe sont",
            "retirés de l'entrée de la policy. Ces clips ne montrent pas une caméra",
            "physiquement débranchée ni l'entrée perturbée fournie au modèle.",
            "",
            f"La {demo} affiche aussi la représentation effective XYZRGB :",
            "RGB brut en haut, points retenus en bas. Elle rend visible la suppression",
            "de l'entrée fixe. Les verdicts ci-dessous proviennent des rollouts et traces",
            "JSON complets, plutôt que de l'apparence du clip.",
            "",
            *entries,
            "",
            "[Rapport et résultats agrégés](report.md) · [Reproduction](reproduction.md).",
            "",
        ]
    )
    write(ROOT / "docs/v4/videos.md", content)


def pilot_text() -> str:
    lines = []
    directory = ROOT / "results/v4/validation"
    for path in sorted(directory.glob("*.json")):
        result = read(path)
        if not isinstance(result, dict) or "rollouts" not in result:
            continue
        grouped = collections.defaultdict(list)
        for row in result["rollouts"]:
            grouped[
                (row.get("group", "non indiqué"), row.get("condition", "non indiqué"))
            ].append(row)
        for (group, condition), rows in sorted(grouped.items()):
            physical = sum(bool(r["physical_success"]) for r in rows)
            strict = sum(bool(r["strict_v2_success"]) for r in rows)
            seeds = sorted({r.get("training_seed") for r in rows}, key=str)
            lines.append(
                f"| [{path.stem}](../../{relative(path)}) | {group} | {condition} | "
                f"{','.join(map(str, seeds))} | {physical}/{len(rows)} | {strict}/{len(rows)} |"
            )
    if not lines:
        return (
            "Aucun résultat pilote au schéma `identity/rollouts` n'est disponible sous "
            "`results/v4/validation/` au moment de cette génération. Le rapport ne lui "
            "attribue aucun score ; vérifier le journal et les fichiers du pilote avant livraison."
        )
    return "\n".join(
        [
            "Ces observations sont de la validation connue, utilisée avant le gel. Elles sont",
            "présentées séparément du nouveau test et ne constituent pas un second test indépendant.",
            "",
            "| Fichier brut | Groupe | Condition | Seed(s) | Physique | Strict V2 |",
            "|---|---|---|---|---:|---:|",
            *lines,
        ]
    )


def documentation(summary: dict, plan: dict) -> None:
    results = summary["groups"]
    contrast = summary["paired_contrasts"]
    main_table = [
        "| Recette | Nominal physique | Nominal strict V2 | Altérations : physique | Altérations : strict V2 |",
        "|---|---:|---:|---:|---:|",
    ]
    readme_table = ["| Recette | Nominal | Neuf altérations |", "|---|---:|---:|"]
    for group in GROUPS:
        cells = results[group]
        main_table.append(
            f"| {LABELS[group]} | {metric_text(cells['nominal'])} | "
            f"{metric_text(cells['nominal'], METRICS[1])} | "
            f"{metric_text(cells['mean_perturbed'])} | "
            f"{metric_text(cells['mean_perturbed'], METRICS[1])} |"
        )
        readme_table.append(
            f"| {LABELS[group]} | {metric_text(cells['nominal'])} | {metric_text(cells['mean_perturbed'])} |"
        )
    paired_table = [
        "| Comparaison, première − seconde | Nominal physique, pp [IC95] | Altérations physique, pp [IC95] | Altérations strict V2, pp [IC95] |",
        "|---|---:|---:|---:|",
    ]
    interpretations = []
    for first, second in CONTRASTS:
        key = first + "_minus_" + second
        value = contrast[key]
        paired_table.append(
            f"| {LABELS[first]} − {LABELS[second]} | "
            f"{difference_text(value['nominal'][METRICS[0]])} | "
            f"{difference_text(value['mean_perturbed'][METRICS[0]])} | "
            f"{difference_text(value['mean_perturbed'][METRICS[1]])} |"
        )
        if first.endswith("aug") and second.endswith("clean"):
            nominal = value["nominal"][METRICS[0]]["estimate_percent_or_pp"]
            robust = value["mean_perturbed"][METRICS[0]]["estimate_percent_or_pp"]
            lo, hi = value["mean_perturbed"][METRICS[0]]["descriptive_95_percent_or_pp"]
            uncertainty = (
                "L'intervalle descriptif recouvre zéro."
                if lo <= 0 <= hi
                else "L'intervalle descriptif reste du même côté de zéro dans ce test."
            )
            interpretations.append(
                f"Pour {LABELS[first].split(',')[0].lower()}, les augmentations changent "
                f"le taux physique nominal de {nominal:+.2f} pp et la moyenne altérée de "
                f"{robust:+.2f} pp. {uncertainty} La conservation nominale n'est pas "
                "démontrée par un test de non-infériorité."
            )
    condition_tables = []
    failure_tables = []
    for condition in CONDITIONS:
        condition_tables.append(
            f"| {LABELS[condition]} | "
            + " | ".join(metric_text(results[g][condition]) for g in GROUPS)
            + " |"
        )
        for group in GROUPS:
            cell = results[group][condition]
            failures = (
                "; ".join(
                    f"{stage}: {count}"
                    for stage, count in sorted(cell["physical_failure_stages"].items())
                )
                or "aucun échec physique"
            )
            failure_tables.append(
                f"| {LABELS[group]} | {LABELS[condition]} | {failures} | {cell['posture_only_failures']} |"
            )
    video_paths = sorted((ROOT / "artifacts/v4/demo").rglob("*.mp4"))
    video = (
        available_link("Vidéo de démonstration locale", video_paths[0])
        if video_paths
        else None
    )
    checkpoint = available_link(
        "Checkpoint local", ROOT / "artifacts/v4/demo/checkpoint.pt"
    )
    demo_links = " · ".join(x for x in (video, checkpoint) if x)
    if not demo_links:
        demo_links = "La démo n'est pas encore présente sous `artifacts/v4/demo/` ; vérifier la livraison avant de présenter ce rapport."
    protocol_path = ROOT / "configs/v4/final_protocol.json"
    protocol_link = available_link("protocole gelé", protocol_path)
    protocol_text = (
        f"Le {protocol_link} fixe recettes, checkpoints, perturbations et scènes. "
        "La vérification indépendante doit confirmer son horodatage antérieur au premier accès au test."
        if protocol_link
        else "Le protocole gelé n'est pas disponible au moment de la génération ; la livraison reste à vérifier."
    )
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
    parity_path = ROOT / "results/v4/v3_clean_trace_parity.json"
    parity = read(parity_path)
    if parity["compared_rollouts"] != 120 or parity["exact_full_traces"] != 120:
        raise ValueError(
            "V3/V4 clean-control trace parity differs from the verified120 rollouts"
        )
    report = "\n".join(
        [
            "# GeoPolicy-Bench V4 — robustesse RGB-D",
            "",
            "Cette étude compare caméra fixe et fusion fixe+poignet, chacune avec et sans",
            "augmentations ciblées, sur un cube, un bac et une consigne fixe. Les résultats",
            "ci-dessous sont calculés exclusivement à partir des rollouts réellement exécutés.",
            "",
            *main_table,
            "",
            "La moyenne altérée attribue le même poids à neuf réglages de perturbation.",
            "Elle décrit ce banc synthétique ; elle ne correspond pas à la fréquence de",
            "pannes d'un capteur réel. Les 540 rollouts par groupe répètent les mêmes",
            "20 scènes, trois seeds et neuf conditions ; ce ne sont pas 540 scènes indépendantes.",
            "",
            "## Avant/après contrôlé et incertitude",
            "",
            *paired_table,
            "",
            *interpretations,
            "",
            "Les intervalles sont un bootstrap apparié croisé à 95 % sur trois seeds et",
            "20 scènes, avec 10 000 tirages (RNG 81). La même scène et ses conditions restent",
            "appariées dans chaque tirage. Pour la moyenne altérée, les neuf conditions sont",
            "des réglages fixes : on ne les rééchantillonne pas comme des scènes indépendantes.",
            "Trois seeds donnent une estimation fragile de la variance d'entraînement ; les",
            "comparaisons sont descriptives, sans correction de multiplicité. Un intervalle",
            "nul au plafond ne prouve pas l'absence d'échecs inconnus. Un intervalle qui",
            "recouvre zéro n'établit ni équivalence de fusion/fixe, ni conservation nominale.",
            "",
            "Les scores V3 historiques ont été mesurés sur un autre test et restent conservés.",
            "On ne soustrait pas leurs taux à V4 pour prétendre à une amélioration appariée.",
            "",
            "## Données, protocole et contrats",
            "",
            "La tâche simple V3 comprend **80 collectes d'entraînement, dont 79 démonstrations",
            "réellement utilisées**. La collecte 10032 est exclue pour échec de prise ; les",
            "79 trajectoires continuées comprennent 9 986 observations/actions, et dix",
            "démonstrations de validation enregistrée sont séparées. Les identités historiques",
            "et `--limit 80` désignent une collecte ou un plafond, pas 80 succès d'entraînement.",
            "Les quatre variantes réutilisent ce sous-ensemble, les actions et normalisations",
            "V3 ; les budgets effectifs et distributions d'augmentations sont dans le plan.",
            "",
            "La reprise du contrôle V3 a été vérifiée sur **120/120 traces entièrement",
            "identiques** : checkpoints V3 diagnostiqués et contrôles V4 sans augmentation",
            "seed 0, deux vues × six conditions × dix scènes pilotes 210000–210009.",
            "Actions, physique et évaluateur concordent exactement dans les JSON conservés.",
            "Ce contrôle réutilise les rollouts déjà exécutés ; il n'ajoute aucun calcul test",
            "et n'établit l'équivalence que pour ces 120 rollouts, pas pour toutes les seeds",
            "ou scènes. [Preuve et portée](../../results/v4/v3_clean_trace_parity.json).",
            "",
            "Les policies prédisent les sept commandes ; le modèle reçoit uniquement XYZRGB,",
            "masques, état robot causal et labels fixes. Poses d'objets et phase teacher servent",
            "à la collecte ou à l'évaluateur. Aucun oracle ne remplace une action étudiante.",
            "Le prior couleur manuel V3 est conservé : cette étude ne l'ablative pas à nouveau.",
            "",
            protocol_text,
            "Le nouveau test est 500000–500019 : 4 groupes × 3 seeds × 10 conditions ×",
            "20 scènes = 2 400 rollouts. Le générateur de rapport refuse toute cellule manquante",
            "ou dupliquée et refuse des entrées effectives initiales incompatibles. Les observations",
            "initiales propres et perturbées, par modalité et avant sélection de vue, sont",
            "conservées par l'évaluateur ; les hashes vérifiés figurent dans",
            "[report_sensor_identity.json](../../results/v4/report_sensor_identity.json).",
            "",
            "**Exception brute conservée :** sur 2 400 initialisations, une composante de",
            "l'image RGB fixe varie de 133 à 134 (8 bits), scène 500017, fusion augmentée",
            "seed 1, points manquants 70 %. La cause n'est pas démontrée. Le contrôle brut",
            "des images reste donc négatif. Le QC après test compare toutes les arrays :",
            "points XYZRGB propres/altérés, masques, état robot, profondeur, calibration et",
            "transformations sont exactement identiques avant sélection de vue. Le modèle",
            "reçoit ces points et l'état, pas l'image RGB complète. L'appariement des entrées",
            "effectives est vérifié sans tolérance ; tous les scores originaux restent dans",
            "les comparaisons, sans exclusion, réévaluation ni modification du protocole.",
            "[QC des modalités et détails de l'exception](../../results/v4/initial_sensor_qc.json).",
            "",
            "Le placement physique exige le cube entièrement dans le bac, une libération",
            "constatée, aucun contact de doigts et une seconde de stabilité après libération,",
            "avec les seuils V3 inchangés. Le critère strict V2 demande en plus la pince ouverte",
            "durant ce délai. Le succès est latched ; il ne prouve pas une stabilité indéfinie.",
            "",
            "### Plan exécuté",
            "",
            "```json",
            plan_text,
            "```",
            "",
            "## Pilote et validation, séparés du test",
            "",
            pilot_text(),
            "",
            "Les observations pilotes peuvent motiver le choix d'une recette. Elles ne doivent",
            "pas être ajoutées au dénominateur du test, et une baisse de loss n'est pas une",
            "preuve de placement. Le journal de sélection distingue hypothèses et causes établies.",
            "",
            "## Courbes réussite / intensité",
            "",
            "![Placement physique](figures/robustness_physical.png)",
            "",
            "![Placement strict V2](figures/robustness_strict.png)",
            "",
            "| Condition | Fixe sans augmentation | Fixe augmentée | Fusion sans augmentation | Fusion augmentée |",
            "|---|---:|---:|---:|---:|",
            *condition_tables,
            "",
            "L'extrémité 100 % de la courbe d'occultation représente la caméra fixe absente.",
            "C'est une condition distincte ; l'interpolation graphique ne prouve pas un continuum",
            "physique de panne. Le bruit et les points manquants concernent les deux vues,",
            "alors que l'occultation et l'absence concernent la caméra fixe.",
            "",
            "## Diagnostics d'échec",
            "",
            "| Recette | Condition | Étapes des échecs physiques | Physique réussi, strict échoué |",
            "|---|---|---|---:|",
            *failure_tables,
            "",
            "Les étapes sont des heuristiques de trace : approche, prise, transport, libération",
            "ou stabilité. Leur fréquence ne démontre pas une cause unique. Une panne de caméra",
            "peut déplacer toute la trajectoire ; une corrélation avec une étape de prise n'isole",
            "pas à elle seule un défaut de perception ou de commande de pince. Les traces et",
            "actions brutes permettent de contrôler les hypothèses sans réécrire les résultats.",
            "",
            "## Démonstration et reproduction",
            "",
            demo_links,
            "",
            "[Index des huit captures préspécifiées, succès et échecs](videos.md).",
            "Les captures de test montrent le RGB brut ; la démo principale montre aussi",
            "l'entrée XYZRGB effective, distinction expliquée dans cet index.",
            "",
            "[Reproduction](reproduction.md) · [CSV des rollouts](../../results/v4/rollouts.csv) ·",
            "[Synthèse et intervalles](../../results/v4/summary.json) · [Bruts test](../../results/v4/test/).",
            "[Vérification finale et mesures de ressources](../../results/v4/delivery_verification.json).",
            "",
            "## Limites",
            "",
            "Les perturbations sont synthétiques, appliquées après acquisition et échantillonnage",
            "des points. Elles modélisent une dégradation de l'entrée de la policy et ne reproduisent",
            "pas toutes les pannes d'un RGB-D réel : trous liés au matériau, pixels volants, biais",
            "temporels, latence, désynchronisation, erreurs de calibration et diffusion infrarouge.",
            "Le bruit axial ne fait pas apparaître des surfaces éliminées avant l'échantillonnage ;",
            "la suppression de points ne correspond pas nécessairement au même pourcentage de",
            "pixels profondeur absents. Les contrôles doivent vérifier cette distinction et les",
            "taux effectifs, particulièrement après recadrage et fusion à budget de points constant.",
            "",
            "Objets/couleurs et consigne sont connus ; la calibration et la dynamique restent",
            "simulées. Aucune variation de tâche, compétence multicible, compréhension de langage",
            "libre ou **sim-to-real réel** n'a été évalué. La comparaison n'impose aucun gain de",
            "fusion ou d'augmentation. Les résultats négatifs sont conservés. V1/V2/V3 restent",
            "séparées ; cette extension locale n'a pas modifié de CV ni publié de contenu en ligne.",
            "",
        ]
    )
    write(ROOT / "docs/v4/report.md", report)
    readme = "\n".join(
        [
            "# GeoPolicy-Bench V4 — robustesse de policies RGB-D",
            "",
            "**Question :** caméra fixe ou fusion fixe+poignet résistent-elles à l'occultation,",
            "à l'absence de caméra, au bruit de profondeur et aux points manquants, et les",
            "augmentations aident-elles sans sacrifier la performance nominale ?",
            "",
            "**Résultats exécutés :** un cube, un bac, consigne fixe ; 20 nouvelles scènes ×",
            "3 seeds, mêmes 79 démonstrations retenues sur 80 collectes, données/actions/budgets",
            "appariés. Les valeurs altérées moyennent neuf perturbations synthétiques fixes.",
            "",
            *readme_table,
            "",
            "Placement physique stable après libération ; les deux critères coïncident dans ce test.",
            "La fusion sans augmentation conserve le meilleur taux observé sur ce banc.",
            "La recette d'augmentation n'apporte pas de gain global démontré : les intervalles",
            "appariés de ses deux contrastes moyens recouvrent zéro. Détails dans le rapport.",
            "",
            demo_links,
            "",
            "[Huit captures préspécifiées avec verdicts réels](videos.md) : RGB brut du",
            "simulateur ; la démo principale montre aussi les points fournis à la policy.",
            "",
            "![Courbes de robustesse physique](figures/robustness_physical.png)",
            "",
            "[Rapport et intervalles](report.md) · [Reproduction](reproduction.md) ·",
            "[Résultats bruts](../../results/v4/test/) · [Code V4](../../src/geopolicy/v4/) ·",
            "[Paramètres centraux](../../configs/v4/plan.json).",
            "",
            "**QC :** les entrées effectives des policies sont identiques entre variantes.",
            "Une variation d'un niveau sur une composante RGB brute est conservée et",
            "[documentée dans le rapport](report.md), avec tous les résultats originaux.",
            "",
            "Pour rejouer la démo depuis le projet :",
            "",
            "```powershell",
            ".venv\\Scripts\\python.exe scripts/v4/demo.py replay",
            "```",
            "",
            "Le checkpoint compact et le [bundle autonome](../../artifacts/v4/demo_bundle.zip)",
            "se passent des données d'entraînement. Reconstruction des résultats :",
            "`python scripts/v4/report.py`.",
            "",
            "**Limites :** perturbations après échantillonnage, RGB-D simulé/calibration idéale,",
            "prior couleur manuel, objets connus et tâche simple. Trois seeds et 20 scènes limitent",
            "les conclusions ; aucun transfert vers un robot réel ou gain universel de fusion",
            "n'est revendiqué. Les échecs et les versions V1/V2/V3 sont conservés séparément.",
            "",
        ]
    )
    write(ROOT / "docs/v4/README.md", readme)
    reproduction = "\n".join(
        [
            "# Reproduire GeoPolicy-Bench V4",
            "",
            "Depuis `C:\\Users\\Alec\\Documents\\ChatGPT\\GeoPolicy-Bench`, utiliser l'environnement",
            "local validé et une seule tâche lourde à la fois sur la RTX 4060 Laptop 8 Go.",
            "Les versions exactes et limites de ressources sont consignées dans le protocole",
            "gelé et le [plan](../../configs/v4/plan.json). Les checkpoints d'optimisation servent",
            "à reprendre les calculs ; le checkpoint compact de démonstration sert à l'inférence.",
            "",
            "## Vérifier les résultats conservés",
            "",
            "```powershell",
            ".venv\\Scripts\\python.exe scripts/v4/report.py",
            "```",
            "",
            "Cette commande ne lance aucune simulation ni entraînement. Elle exige les 2 400",
            "rollouts test et le QC `initial_sensor_qc.json`, vérifie l'appariement de chaque",
            "entrée effective initiale, puis reconstruit",
            "CSV, synthèse, courbes et documentation. Une erreur de complétude ou d'identité",
            "interrompt la génération ; elle n'invente, ne filtre et ne remplace aucun résultat.",
            "Le QC peut être recalculé avec `python scripts/v4/sensor_qc.py`. Il conserve",
            "l'exception RGB brute et exige l'égalité exacte de toutes les entrées du modèle.",
            "",
            "## Démo avec checkpoint local, sans données d'entraînement",
            "",
            "```powershell",
            ".venv\\Scripts\\python.exe scripts/v4/demo.py replay",
            "```",
            "",
            "Sorties : `artifacts/v4/demo/reproduced/` (MP4, GIF, trace, métriques et",
            "vérification exacte des observations initiales, actions et physique). Le modèle",
            "est fusion augmentée, seed 0, scène 500000, caméra fixe absente. Cette scène est",
            "la première du test et a été choisie avant son observation, succès ou échec.",
            "La vidéo distingue RGB physique du simulateur et représentation XYZRGB réellement",
            "fournie à la policy. Le bundle local `artifacts/v4/demo_bundle.zip` comprend le",
            "checkpoint compact, la source, les paramètres et les traces attendues. Après",
            "extraction dans un dossier indépendant, depuis ce dossier :",
            "",
            "```powershell",
            "python scripts/v4/demo.py replay",
            "```",
            "",
            "Utiliser Python 3.11.9 et les versions verrouillées dans `requirements-lock.txt`.",
            "La livraison a aussi exécuté ce replay avec une copie isolée de la source et",
            "un second environnement local épinglé ; voir `delivery_verification.json`.",
            "",
            "## Reprise et reproduction des calculs",
            "",
            "Les phases sérielles conservent leurs logs et index terminés sous `artifacts/v4/jobs/`.",
            "Une phase déjà terminée ne relance pas ses cellules. Après interruption :",
            "",
            "```powershell",
            ".venv\\Scripts\\python.exe scripts/v4/run_jobs.py configs/v4/main_train_jobs.json",
            ".venv\\Scripts\\python.exe scripts/v4/run_jobs.py configs/v4/confirmation_jobs.json",
            ".venv\\Scripts\\python.exe scripts/v4/run_jobs.py configs/v4/final_jobs.json",
            "```",
            "",
            "Le contrôleur ajoute `--resume` lorsqu'un checkpoint existe, sauvegarde chaque",
            "rollout et vérifie les limites du PC avant chaque processus lourd. Le plafond",
            "temporel global est enregistré dans `artifacts/v4/budget.json` ; son expiration",
            "arrête les nouvelles cellules et conserve les résultats. Un nouveau budget ne",
            "doit pas être ajouté silencieusement pour prolonger l'étude livrée.",
            "",
            "Dans une copie indépendante du projet, l'entraînement exige les 79 HDF5 train",
            "retenus **et les 10 HDF5 de validation enregistrée**, leurs manifests sous",
            "`configs/v3/`, ainsi que `artifacts/v3/runs/bc_continued_history480_s0/best.pt`",
            "pour reprendre exactement la normalisation V3. La démo compacte se passe de",
            "ces données et de ce checkpoint de normalisation. Sans anciens runs V4 dans",
            "cette copie, une cellule d'entraînement au même budget se lance ainsi :",
            "",
            "```powershell",
            ".venv\\Scripts\\python.exe -m geopolicy.v4 preflight",
            ".venv\\Scripts\\python.exe -m geopolicy.v4 train --scope main --view fusion --augmented --seed 0",
            "```",
            "",
            "Omettre `--augmented` pour le contrôle et choisir `--view fixed` pour la caméra",
            "fixe. Les seeds sont 0, 1 et 2. Les paramètres centraux restent ceux du plan gelé.",
            "Pour une reproduction complète du test connu, conserver les 12 checkpoints et le protocole",
            "gelé dans cette copie, sans fichiers de résultats/index V4 antérieurs, puis utiliser",
            "`configs/v4/final_jobs.json`. Toute modification de recette requiert un autre test.",
            "",
            "## Contrats d'une reproduction",
            "",
            "Conserver les SHA des 79 démonstrations V3, le même ordre d'épisodes et RNG de",
            "batches, le même budget et la normalisation V3. Une augmentation a son propre RNG",
            "reprenable ; elle ne doit pas changer les tirages d'actions démontrées. Conserver",
            "les observations initiales propres/altérées des deux caméras avant sélection de vue.",
            "Les scènes 500000–500019 sont désormais connues : elles servent au replay, plus au",
            "choix d'une nouvelle amélioration. Toute recette ultérieure a besoin d'un autre test.",
            "",
            "Ne pas regénérer le protocole après accès au test. Choisir de nouveaux espaces V4",
            "pour une reproduction distincte ; ne pas remplacer les bruts ni poids livrés.",
            "Vérifier les commandes de calcul/démo dans le journal de livraison et la CLI V4",
            "avant de lancer un entraînement ; les résultats de ce rapport ne prouvent pas",
            "une reproduction sur un autre matériel ou une égalité bit à bit interplateforme.",
            "",
            demo_links,
            "",
            "[Rapport](report.md) · [Synthèse JSON](../../results/v4/summary.json) ·",
            "[CSV complet](../../results/v4/rollouts.csv). Aucun artefact V1/V2/V3 n'est réécrit",
            "par ce générateur ; aucun CV n'est modifié.",
            "",
        ]
    )
    write(ROOT / "docs/v4/reproduction.md", reproduction)


def main() -> None:
    plan_path = ROOT / "configs/v4/plan.json"
    protocol_path = ROOT / "configs/v4/final_protocol.json"
    if not plan_path.exists() or not protocol_path.exists():
        raise ValueError(
            "V4 plan and frozen final protocol are required before final reporting"
        )
    plan = read(plan_path)
    if (
        plan["reserved_test_scenes"] != [SCENES[0], SCENES[-1]]
        or plan["training_seeds"] != list(SEEDS)
        or set(plan["conditions"]) != set(CONDITIONS)
    ):
        raise ValueError("Report contract differs from the executed V4 plan")
    evidence = training_evidence(plan)
    rows, inputs = load_rows(read(protocol_path))
    identity = verify_identity(rows)
    summary = summarize(rows, inputs, identity)
    summary["plan_sha256"] = sha(plan_path)
    summary["protocol_sha256"] = sha(protocol_path)
    summary["report_source_sha256"] = sha(Path(__file__))
    summary["training_evidence"] = evidence
    write(ROOT / "results/v4/summary.json", summary)
    fieldnames = sorted({key for row in rows for key in row})
    csv_path = ROOT / "results/v4/rollouts.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    write(
        ROOT / "results/v4/report_completeness.json",
        {"complete": True, "expected_rollouts": 2400, "observed_rollouts": len(rows)},
    )
    figures(summary)
    video_index(rows)
    documentation(summary, plan)
    print(
        json.dumps(
            {
                "rollouts": len(rows),
                "all_initial_pairs_exact": identity["all_initial_pairs_exact"],
                "summary": "results/v4/summary.json",
                "report": "docs/v4/report.md",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
