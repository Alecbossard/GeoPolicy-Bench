"""Validate completed V4 confirmation before freezing a new reserved test.

Only stdlib/NumPy and the NumPy-only sensor corruption module are imported.
No simulator, Torch, GPU, checkpoint loading, or policy selection from test.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from geopolicy.v4.perturbations import (
    CAMERAS,
    evaluation_rng,
    packet_digest,
    perturb,
)

GROUPS = ("fixed_clean", "fixed_aug", "fusion_clean", "fusion_aug")
SEEDS = (0, 1, 2)
SCENES = tuple(range(210100, 210110))
CONDITIONS = ("nominal", "occlusion60", "absent", "depth010", "depth025", "missing70")
METRICS = ("physical_success", "strict_v2_success")
CONTRASTS = (
    ("fixed_aug", "fixed_clean"),
    ("fusion_aug", "fusion_clean"),
    ("fusion_clean", "fixed_clean"),
    ("fusion_aug", "fixed_aug"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write(path, value):
    if path not in (
        ROOT / "results/v4/validation_decision.json",
        ROOT / "docs/v4/validation.md",
    ):
        raise ValueError(f"Unauthorized validation output: {path}")
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkpoints(plan):
    manifest_path = ROOT / "configs/v3/single_dataset.json"
    selected = [
        r for r in read(manifest_path) if r["split"] == "train" and r["success"]
    ]
    expected_data = [
        {"path": r["path"], "sha256": r["sha256"], "frames": r["frames"]}
        for r in selected
    ]
    require(
        len(expected_data) == 79,
        "V3 training subset is not79 successful demonstrations",
    )
    require(
        sum(r["frames"] for r in expected_data) == 9986,
        "Continued training frame count changed",
    )
    require(
        not any(r["scene_seed"] == 10032 for r in selected),
        "Failed grasp collection entered training",
    )
    for item in expected_data:
        require(
            sha(ROOT / item["path"]) == item["sha256"],
            f"Demonstration changed: {item['path']}",
        )
    records = {}
    inputs = {rel(manifest_path): sha(manifest_path)}
    reference_config = None
    for group in GROUPS:
        view, augmentation = group.split("_")
        for seed in SEEDS:
            name = f"{group}_s{seed}"
            path = ROOT / f"artifacts/v4/main_runs/{name}/manifest.json"
            manifest = read(path)
            cfg = manifest["config"]
            require(
                manifest["completed_update"] == 2000
                and cfg["updates"] == plan["training"]["updates"] == 2000
                and manifest["actual_train_episodes"] == 79
                and manifest["training_frames"] == 9986
                and manifest["validation_episodes"] == 10
                and manifest["training_data"] == expected_data,
                f"Incomplete or unmatched training data/budget: {name}",
            )
            require(
                cfg["view"] == view
                and cfg["augmented"] is (augmentation == "aug")
                and cfg["seed"] == seed
                and cfg["augmentation_recipe"] == plan["augmentation"],
                f"Checkpoint labels or recipe differ: {name}",
            )
            comparable = {
                k: v for k, v in cfg.items() if k not in ("view", "augmented", "seed")
            }
            if reference_config is None:
                reference_config = comparable
            require(
                comparable == reference_config,
                f"Unmatched main training configuration: {name}",
            )
            checkpoint = ROOT / f"artifacts/v4/main_runs/{name}/best.pt"
            digest = sha(checkpoint)
            require(
                digest == manifest["best_checkpoint_sha256"],
                f"Checkpoint SHA differs from manifest: {name}",
            )
            records[(group, seed)] = {
                "checkpoint": rel(checkpoint),
                "sha256": digest,
                "manifest_sha256": sha(path),
                "completed_update": manifest["completed_update"],
            }
            inputs[rel(path)] = sha(path)
    checks_path = ROOT / "results/v4/input_checks.json"
    checks = read(checks_path)
    require(
        checks["dataset_episodes"] == 79 and checks["dataset_frames"] == 9986,
        "Input-check dataset identity changed",
    )
    for key in (
        "all_nominal_cached_inputs_exact_v3",
        "recorded_live_exact",
        "axial_depth_ray_equivalence",
        "nested_missing_masks",
        "finite_empty_policy",
        "cuda_augmented_resume_next_update_exact",
    ):
        require(checks[key] is True, f"Input contract failed: {key}")
    inputs[rel(checks_path)] = sha(checks_path)
    return records, inputs


def confirmation(plan, records):
    paths = sorted((ROOT / "results/v4/validation").glob("confirmation_*.json"))
    require(
        len(paths) == 72,
        f"Confirmation requires72 files of10 episodes, found{len(paths)}",
    )
    rows = {}
    inputs = {}
    sources = None
    for path in paths:
        value = read(path)
        identity, rollouts = value["identity"], value["rollouts"]
        require(
            len(rollouts) == 10,
            f"Incomplete confirmation file: {rel(path)} ({len(rollouts)}/10)",
        )
        require(
            identity["split"] == "validation"
            and identity["first"] == SCENES[0]
            and identity["episodes"] == len(SCENES)
            and identity["condition"] in CONDITIONS
            and identity["perturbation"] == plan["conditions"][identity["condition"]]
            and identity["plan_sha256"] == sha(ROOT / "configs/v4/plan.json"),
            f"Unexpected confirmation identity: {rel(path)}",
        )
        require(
            sorted(r["scene_seed"] for r in rollouts) == list(SCENES),
            f"Confirmation scene IDs differ: {rel(path)}",
        )
        if sources is None:
            sources = identity["sources"]
        require(
            identity["sources"] == sources,
            f"Mixed confirmation source versions: {rel(path)}",
        )
        for row in rollouts:
            key = (
                row["group"],
                row["condition"],
                row["training_seed"],
                row["scene_seed"],
            )
            require(key not in rows, f"Duplicate confirmation rollout: {key}")
            record = records.get((row["group"], row["training_seed"]))
            require(record is not None, f"Unexpected group/seed: {key}")
            require(
                row["condition"] == identity["condition"]
                and identity["checkpoint"] == record["checkpoint"]
                and identity["checkpoint_sha256"]
                == row["checkpoint_sha256"]
                == record["sha256"],
                f"Unmatched checkpoint labels/SHA: {key}",
            )
            require(
                all(type(row[m]) is bool for m in METRICS), f"Nonboolean success: {key}"
            )
            require(
                not row.get("diagnostic_oracle", False),
                f"Oracle student rollout: {key}",
            )
            rows[key] = (row, path.stem)
        inputs[rel(path)] = sha(path)
    expected = {
        (group, condition, seed, scene)
        for group in GROUPS
        for condition in CONDITIONS
        for seed in SEEDS
        for scene in SCENES
    }
    require(
        set(rows) == expected,
        f"Missing/unexpected confirmation cells: {sorted(expected-set(rows))}, {sorted(set(rows)-expected)}",
    )
    return rows, inputs, sources


def sensor_and_action_checks(rows, plan):
    clean_by_scene, corrupted_by_scene_condition = {}, {}
    mismatch = []
    steps = 0
    for key in sorted(rows):
        row, name = rows[key]
        scene, condition = row["scene_seed"], row["condition"]
        directory = ROOT / f"artifacts/v4/evaluations/{name}"
        initial_path = directory / f"initial/{scene}.npz"
        require(
            sha(initial_path) == row["initial_npz_sha256"],
            f"Initial observation NPZ changed: {key}",
        )
        with np.load(initial_path, allow_pickle=False) as data:
            packet = {
                camera: {
                    field: data[f"{camera}__clean__{field}"]
                    for field in (
                        "rgb",
                        "depth_m",
                        "intrinsic",
                        "base_from_camera",
                        "points",
                        "mask",
                    )
                }
                for camera in CAMERAS
            }
            clean = hashlib.sha256(
                (
                    packet_digest(
                        packet,
                        (
                            "rgb",
                            "depth_m",
                            "intrinsic",
                            "base_from_camera",
                            "points",
                            "mask",
                        ),
                    )
                    + hashlib.sha256(data["state"].tobytes()).hexdigest()
                ).encode()
            ).hexdigest()
            changed = perturb(
                packet, plan["conditions"][condition], evaluation_rng(scene, 0)
            )
            for camera in CAMERAS:
                for field in ("points", "mask"):
                    require(
                        np.array_equal(
                            changed[camera][field],
                            data[f"{camera}__corrupted__{field}"],
                        ),
                        f"Initial perturbation mismatch: {key}, {camera}/{field}",
                    )
            corrupted = packet_digest(changed)
        require(
            clean == row["initial_clean_hash"]
            and corrupted == row["initial_corrupted_hash"],
            f"Initial hashes do not match saved modalities: {key}",
        )
        for field, digest, mapping, identity in (
            ("clean", clean, clean_by_scene, scene),
            ("corrupted", corrupted, corrupted_by_scene_condition, (scene, condition)),
        ):
            if identity in mapping and mapping[identity] != digest:
                mismatch.append(
                    {
                        "cell": key,
                        "field": field,
                        "reference": mapping[identity],
                        "observed": digest,
                    }
                )
            else:
                mapping[identity] = digest
        trace = read(directory / f"traces/{scene}.json")
        require(
            len(trace) == row["steps"] and len(trace) > 0,
            f"Missing action trace: {key}",
        )
        for step in trace:
            action = np.asarray(step["action"], dtype=float)
            require(
                action.shape == (7,)
                and np.isfinite(action).all()
                and (np.abs(action) <= 1.000001).all(),
                f"Invalid/clipping action trace: {key}",
            )
        steps += len(trace)
    require(not mismatch, "Unmatched initial sensor packets: " + json.dumps(mismatch))
    return {
        "initial_clean_pairs_exact": True,
        "initial_corrupted_pairs_exact": True,
        "saved_initial_npz_recomputed": len(rows),
        "unique_clean_scenes": len(clean_by_scene),
        "unique_corrupted_scene_conditions": len(corrupted_by_scene_condition),
        "action_dimensions_finiteness_clipping_verified": True,
        "action_steps_checked": steps,
        "mismatches": mismatch,
    }


def estimates(rows):
    rng = np.random.default_rng(81)
    seeds = rng.integers(3, size=(10000, 3))
    scenes = rng.integers(10, size=(10000, 10))

    def bootstrap(matrix):
        draws = matrix[seeds[:, :, None], scenes[:, None, :]].mean((1, 2)) * 100
        return {
            "estimate_percent_or_pp": float(matrix.mean() * 100),
            "descriptive_95_percent_or_pp": np.percentile(draws, [2.5, 97.5]).tolist(),
        }

    matrices, scores = {}, {}
    for group in GROUPS:
        matrices[group], scores[group] = {}, {}
        for condition in CONDITIONS:
            matrices[group][condition] = {}
            scores[group][condition] = {"rollouts": 30}
            for metric in METRICS:
                matrix = np.asarray(
                    [
                        [
                            int(rows[(group, condition, seed, scene)][0][metric])
                            for scene in SCENES
                        ]
                        for seed in SEEDS
                    ],
                    dtype=float,
                )
                matrices[group][condition][metric] = matrix
                scores[group][condition][metric] = {
                    "successes": int(matrix.sum()),
                    "per_seed_successes": matrix.sum(1).astype(int).tolist(),
                    **bootstrap(matrix),
                }
        matrices[group]["mean_perturbed"] = {
            metric: np.stack(
                [matrices[group][condition][metric] for condition in CONDITIONS[1:]]
            ).mean(0)
            for metric in METRICS
        }
        scores[group]["mean_perturbed"] = {
            "rollouts": 150,
            "fixed_conditions": list(CONDITIONS[1:]),
            **{
                metric: {
                    "successes": sum(
                        scores[group][condition][metric]["successes"]
                        for condition in CONDITIONS[1:]
                    ),
                    "per_seed_successes": np.stack(
                        [
                            matrices[group][condition][metric].sum(1)
                            for condition in CONDITIONS[1:]
                        ]
                    )
                    .sum(0)
                    .astype(int)
                    .tolist(),
                    **bootstrap(matrices[group]["mean_perturbed"][metric]),
                }
                for metric in METRICS
            },
        }
    contrasts = {
        first
        + "_minus_"
        + second: {
            condition: {
                metric: bootstrap(
                    matrices[first][condition][metric]
                    - matrices[second][condition][metric]
                )
                for metric in METRICS
            }
            for condition in ("nominal", "mean_perturbed")
        }
        for first, second in CONTRASTS
    }
    retention = {}
    for view in ("fixed", "fusion"):
        first, second = view + "_aug", view + "_clean"
        difference = (
            matrices[first]["nominal"][METRICS[0]]
            - matrices[second]["nominal"][METRICS[0]]
        )
        retention[view] = {
            "per_seed_change_successes_out_of_10": difference.sum(1)
            .astype(int)
            .tolist(),
            **contrasts[first + "_minus_" + second]["nominal"],
            "formal_noninferiority_test": False,
            "interpretation": "Observed paired nominal difference; neither a small loss nor an interval overlapping zero proves nominal retention",
        }
    return scores, contrasts, retention


def pilot():
    phases = collections.defaultdict(lambda: {"files": 0, "rollouts": 0})
    paths = sorted((ROOT / "results/v4/validation").glob("pilot*.json"))
    inputs = {}
    for path in paths:
        value = read(path)
        require(
            "identity" in value and "rollouts" in value,
            f"Unexpected pilot result: {rel(path)}",
        )
        phase = (
            "V3 diagnostic"
            if path.stem.startswith("pilot_v3_")
            else path.stem.split("_")[0]
        )
        phases[phase]["files"] += 1
        phases[phase]["rollouts"] += len(value["rollouts"])
        inputs[rel(path)] = sha(path)
    require(
        sum(item["rollouts"] for item in phases.values()) == 420,
        "Pilot raw result count differs from420; inspect before delivery",
    )
    return {
        "rollouts": 420,
        "phases": dict(phases),
        "input_sha256": inputs,
        "decision": read(ROOT / "results/v4/pilot_decision.json"),
    }


def md_interval(value):
    lo, hi = value["descriptive_95_percent_or_pp"]
    return f"{value['estimate_percent_or_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def documentation(decision):
    scores = decision["scores"]
    rows = [
        "# V4 — validation et décision avant le test réservé",
        "",
        "La confirmation comprend **720 rollouts** : quatre groupes × trois seeds ×",
        "six conditions × dix scènes 210100–210109. Elle est disjointe du pilote et",
        "du futur test 500000–500019. Tous les modèles ont terminé le même budget de",
        "2 000 updates sur 79 démonstrations réussies, retenues sur 80 collectes V3.",
        "",
        "| Groupe | Condition | Physique /30 | Physique seeds0,1,2 /10 | Strict V2 /30 |",
        "|---|---|---:|---|---:|",
    ]
    for group in GROUPS:
        for condition in CONDITIONS:
            cell = scores[group][condition]
            rows.append(
                f"| {group} | {condition} | {cell[METRICS[0]]['successes']}/30 | {cell[METRICS[0]]['per_seed_successes']} | {cell[METRICS[1]]['successes']}/30 |"
            )
    rows.extend(
        [
            "",
            "## Comparaisons appariées descriptives",
            "",
            "| Première − seconde | Nominal physique, pp [IC95] | Moyenne cinq altérations physique, pp [IC95] | Moyenne cinq altérations strict, pp [IC95] |",
            "|---|---:|---:|---:|",
        ]
    )
    for first, second in CONTRASTS:
        value = decision["paired_contrasts"][first + "_minus_" + second]
        rows.append(
            f"| {first} − {second} | {md_interval(value['nominal'][METRICS[0]])} | {md_interval(value['mean_perturbed'][METRICS[0]])} | {md_interval(value['mean_perturbed'][METRICS[1]])} |"
        )
    rows.extend(
        [
            "",
            "Bootstrap apparié croisé : trois seeds, dix scènes, 10 000 tirages, RNG81 ;",
            "intervalles descriptifs à95 %, sans correction de multiplicité. Les cinq réglages",
            "altérés sont moyennés à poids égaux dans chaque paire seed/scène avant les tirages.",
            "Les répétitions ne créent pas 150 scènes indépendantes. Au plafond/plancher, un",
            "intervalle nul ne révèle pas les événements non observés. Ce protocole n'établit",
            "pas une non-infériorité nominale et un contraste ambigu ne prouve pas l'équivalence.",
            "",
            "## Pilote conservé séparément :420 rollouts",
            "",
            "| Phase | Fichiers | Rollouts exécutés |",
            "|---|---:|---:|",
        ]
    )
    for phase, value in sorted(decision["pilot"]["phases"].items()):
        rows.append(f"| {phase} | {value['files']} | {value['rollouts']} |")
    rows.extend(
        [
            "",
            "Le diagnostic V3 seed0 compte120 épisodes ; le pilote à1000updates compte60",
            "épisodes sur cinq scènes et un sous-ensemble de conditions ; le pilote complet",
            "à2000updates compte240 épisodes. Ces observations connues ont motivé la comparaison",
            "de la recette A unique. Elles restent exclues des720 rollouts de confirmation.",
            "Au pilote complet, fixe clean/aug est identique et fusion augmentée perd quatre",
            "succès sur les cinquante rollouts altérés seed0. Aucun gain d'augmentation n'y a",
            "été démontré ; les résultats à1000updates et les checkpoints pilotes sont conservés.",
            "",
            "## Décision",
            "",
            "**Procéder au test réservé avec les douze checkpoints et la recette A inchangée.**",
            "Cette décision repose sur la validité des données/entrées, du clipping, de la",
            "complétude et de l'appariement. La matrice négative ou ambiguë demeure une",
            "comparaison contrôlée utile demandée par l'utilisateur ; aucun seuil de succès",
            "ou gain de fusion/augmentation n'est imposé pour publier les résultats locaux.",
            "Aucun gain d'augmentation n'est revendiqué à partir de cette validation limitée.",
            "Aucun nouveau réglage, budget ou choix n'est effectué à partir du futur test.",
            "",
            "Les observations initiales propres/altérées des deux modalités sont sauvegardées",
            "et recomputées ; les sept commandes finies restent bornées à[−1,1]. Les deux vues",
            "ont exactement les mêmes entrées initiales avant sélection de vue. Les perturbations",
            "agissent après recadrage/voxelisation/échantillonnage de512points par caméra ; elles",
            "ne modélisent pas une panne RGB-D physique complète. Aucun sim-to-real réel évalué.",
            "",
            "[Décision et preuves JSON](../../results/v4/validation_decision.json) ·",
            "[Bruts de validation](../../results/v4/validation/) · [Plan](../../configs/v4/plan.json).",
            "",
        ]
    )
    write(ROOT / "docs/v4/validation.md", "\n".join(rows))


def main():
    # An immutable decision referenced by a frozen protocol must never be rewritten.
    require(
        not (ROOT / "configs/v4/final_protocol.json").exists(),
        "Validation decision already frozen; read it without regenerating",
    )
    require(
        not list((ROOT / "results/v4/test").glob("*.json")),
        "Reserved test already accessed; validation decision cannot be regenerated",
    )
    try:
        plan = read(ROOT / "configs/v4/plan.json")
        require(
            plan["training_seeds"] == list(SEEDS)
            and plan["confirmation_scenes"] == [SCENES[0], SCENES[-1]]
            and tuple(plan["pilot_conditions"]) == CONDITIONS,
            "Validation contract differs from plan",
        )
        records, training_inputs = checkpoints(plan)
        rows, validation_inputs, sources = confirmation(plan, records)
        sensor_checks = sensor_and_action_checks(rows, plan)
        scores, contrasts, retention = estimates(rows)
        result = {
            "decided_utc": datetime.now(timezone.utc).isoformat(),
            "proceed_to_reserved_test": True,
            "augmentation_gain_demonstrated_on_validation": False,
            "augmentation_gain_interpretation": "Limited descriptive validation; negative or ambiguous recipes retained without claiming a gain",
            "selection": "Keep all four controlled cells, three seeds, single predefined augmentation recipe A unchanged; no test-informed tuning",
            "selected_augmentation_recipe": plan["augmentation"],
            "scores": scores,
            "paired_contrasts": contrasts,
            "nominal_retention_descriptive": retention,
            "completeness": {
                "files": 72,
                "episodes_per_file": 10,
                "rollouts": 720,
                "training_seeds": 3,
                "unique_scenes": 10,
                "conditions": 6,
            },
            "data_and_checkpoint_checks": {
                "used_train": 79,
                "collected_train": 80,
                "training_frames": 9986,
                "recorded_validation": 10,
                "updates_per_main_model": 2000,
                "matched_data_config_and_budgets": True,
                "checkpoints": {
                    f"{group}_s{seed}": record
                    for (group, seed), record in records.items()
                },
            },
            "sensor_action_checks": sensor_checks,
            "statistics": {
                "method": "paired crossed training-seed/scene bootstrap",
                "draws": 10000,
                "rng_seed": 81,
                "descriptive_level": 0.95,
                "multiplicity_correction": False,
                "condition_resampling": False,
                "formal_noninferiority": False,
            },
            "pilot": pilot(),
            "plan_sha256": sha(ROOT / "configs/v4/plan.json"),
            "confirmation_sources": sources,
            "input_sha256": {**training_inputs, **validation_inputs},
            "helper_sha256": sha(Path(__file__)),
            "reserved_test_accessed": False,
        }
        write(ROOT / "results/v4/validation_decision.json", result)
        documentation(result)
        print(
            json.dumps(
                {
                    "proceed_to_reserved_test": True,
                    "confirmation_rollouts": 720,
                    "pilot_rollouts_separate": 420,
                    "augmentation_gain_demonstrated_on_validation": False,
                }
            ),
            flush=True,
        )
    except Exception as exc:
        write(
            ROOT / "results/v4/validation_decision.json",
            {
                "decided_utc": datetime.now(timezone.utc).isoformat(),
                "proceed_to_reserved_test": False,
                "augmentation_gain_demonstrated_on_validation": False,
                "diagnostic": {"exception": type(exc).__name__, "reason": str(exc)},
                "raw_results_modified": False,
                "reserved_test_accessed": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
