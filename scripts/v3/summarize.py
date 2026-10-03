"""Summaries from executed raw rows only; never changes policies or protocol."""

import collections
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from geopolicy.v3.common import write


def paired_interval(difference):
    rng = np.random.default_rng(81)
    seeds = rng.integers(difference.shape[0], size=(10000, difference.shape[0]))
    scenes = rng.integers(difference.shape[1], size=(10000, difference.shape[1]))
    draws = difference[seeds[:, :, None], scenes[:, None, :]].mean((1, 2)) * 100
    return dict(
        difference_pp=float(difference.mean() * 100),
        descriptive_95_pp=np.percentile(draws, [2.5, 97.5]).tolist(),
    )


def main():
    protocol = json.loads((ROOT / "configs/v3/final_protocol.json").read_text())
    grouped = collections.defaultdict(list)
    reference_hashes = None
    mismatches = []
    sensor_matrices = collections.defaultdict(list)
    flat = []
    for record in protocol["registry"]:
        path = ROOT / f"results/v3/test/final_{record['key']}.json"
        result = json.loads(path.read_text())
        rows = result["rollouts"]
        assert len(rows) == protocol["episodes"]
        assert [r["scene_seed"] for r in rows] == list(
            range(protocol["first"], protocol["first"] + protocol["episodes"])
        )
        assert all(
            r["checkpoint_sha256"] == record["checkpoint_sha256"]
            and not r["diagnostic_oracle"]
            for r in rows
        )
        hashes = [r["initial_sensor_sha256"] for r in rows]
        if reference_hashes is None:
            reference_hashes = hashes
        for row, observed, reference in zip(rows, hashes, reference_hashes):
            if observed != reference:
                mismatches.append(
                    dict(
                        key=record["key"],
                        scene_seed=row["scene_seed"],
                        observed_sha256=observed,
                        reference_sha256=reference,
                    )
                )
        sensor_matrices[record["group"]].append((record["seed"], hashes))
        flat.extend(rows)
        grouped[record["group"]].append((record["seed"], rows))
    if mismatches:
        quality = json.loads(
            (ROOT / "results/v3/sensor_identity_quality_check.json").read_text()
        )
        assert quality["raw_mismatches"] == mismatches
        assert (
            not quality["original_rows_changed"]
            and quality["diagnostic_only_no_tuning"]
        )
    sensors = {
        group: np.asarray([values for _, values in sorted(items)])
        for group, items in sensor_matrices.items()
    }
    summaries, matrices = {}, {}
    for group, items in grouped.items():
        items.sort()
        assert [seed for seed, _ in items] == [0, 1, 2]
        matrices[group] = {
            metric: np.asarray([[int(r[metric]) for r in rows] for _, rows in items])
            for metric in ("physical_success", "strict_v2_success")
        }
        summaries[group] = dict(
            episodes=150,
            per_seed_physical=matrices[group]["physical_success"].sum(1).tolist(),
            per_seed_strict=matrices[group]["strict_v2_success"].sum(1).tolist(),
            physical=int(matrices[group]["physical_success"].sum()),
            strict=int(matrices[group]["strict_v2_success"].sum()),
            failure_stages=dict(
                collections.Counter(
                    r["failure_stage"] for _, rows in items for r in rows
                )
            ),
            posture_only=sum(
                r["posture_only_failure"] for _, rows in items for r in rows
            ),
            collisions=sum(r["collision"] for _, rows in items for r in rows),
            median_policy_p95_ms=float(
                np.median([r["policy_p95_ms"] for _, rows in items for r in rows])
            ),
            median_preprocess_p95_ms=float(
                np.median([r["preprocess_p95_ms"] for _, rows in items for r in rows])
            ),
        )
    contrasts = {}
    for first, second in (
        ("v3_fusion_prior", "v3_original_bc"),
        ("v3_fusion_prior", "v3_wrist_prior"),
        ("v3_fusion_prior", "v3_fixed_prior"),
        ("v3_fusion_no_prior", "v3_fusion_prior"),
        ("v3_original_bc", "v3_v1_recipe"),
    ):
        matched = (sensors[first] == sensors[second]).all(0)
        assert matched.sum() >= 49
        contrasts[first + "_minus_" + second] = {
            metric: paired_interval(
                (matrices[first][metric] - matrices[second][metric])[:, matched]
            )
            for metric in ("physical_success", "strict_v2_success")
        }
        contrasts[first + "_minus_" + second].update(
            matched_scenes_per_seed=int(matched.sum()),
            matched_rollouts=int(matched.sum()) * 3,
            excluded_scene_seeds=[int(400000 + i) for i in np.flatnonzero(~matched)],
        )
    result = dict(
        task="single_cube_single_tray_fixed_instruction",
        new_test_scenes=[400000, 400049],
        rollouts=len(flat),
        groups=summaries,
        contrasts=contrasts,
        all_initial_sensors_exact=not mismatches,
        sensor_identity=dict(
            reference="single_fixed_prior_s0",
            matching_rollouts=len(flat) - len(mismatches),
            total_rollouts=len(flat),
            mismatches=mismatches,
            treatment="All raw scores retained. Paired sensitivity intervals omit each unmatched scene across all three seeds for affected comparisons only, using identity rather than outcome.",
        ),
        uncertainty="Descriptive paired seed/scene bootstrap, three training seeds; no multiplicity correction. Bootstrap at a ceiling cannot capture unobserved failures. No general multi-object, language or hardware claim.",
        historical_transfer="Preserved V1/V2 have different full-task training data and budgets; their test transfer is not a controlled algorithm comparison.",
    )
    write(ROOT / "results/v3/final_summary.json", result)
    with (ROOT / "results/v3/all_test_rollouts.csv").open(
        "w", newline="", encoding="utf8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
