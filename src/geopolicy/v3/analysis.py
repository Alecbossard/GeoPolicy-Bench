"""Paired descriptive validation comparisons; three seeds are a small sample."""
import json
import numpy as np
from .common import ROOT, write


def matrix(variant, split="tuning", metric="physical_success"):
    rows = []
    for seed in (0, 1, 2):
        path = ROOT / f"results/v3/validation/bc_{variant}80_s{seed}_{split}.json"
        records = json.loads(path.read_text())["rollouts"]
        assert len(records) == 20 and all(r["training_seed"] == seed for r in records)
        assert [r["scene_seed"] for r in records] == list(range(records[0]["scene_seed"], records[0]["scene_seed"] + 20))
        rows.append([int(r[metric]) for r in records])
    return np.asarray(rows)


def interval(difference):
    rng = np.random.default_rng(81)
    seeds = rng.integers(3, size=(10000, 3))
    scenes = rng.integers(20, size=(10000, 20))
    draws = difference[seeds[:, :, None], scenes[:, None, :]].mean((1, 2)) * 100
    return dict(difference_pp=float(difference.mean() * 100),
                descriptive_95_pp=np.percentile(draws, [2.5, 97.5]).tolist())


def analyze():
    variants = ["original", "continued", "state_history4", "binary", "continued_history4"]
    results = {}
    for metric in ("physical_success", "strict_v2_success"):
        values = {v: matrix(v, metric=metric) for v in variants}
        summary = {v: dict(successes=int(a.sum()), episodes=60, per_seed=a.sum(1).tolist()) for v, a in values.items()}
        contrasts = {v + "_minus_original": interval(values[v] - values["original"]) for v in variants[1:]}
        contrasts["data_history_interaction"] = interval(values["continued_history4"] - values["state_history4"] - values["continued"] + values["original"])
        results[metric] = dict(summary=summary, contrasts=contrasts)
    identity = []
    for seed in (0, 1, 2):
        reference = json.loads((ROOT / f"results/v3/validation/bc_original80_s{seed}_tuning.json").read_text())["rollouts"]
        for variant in variants[1:]:
            other = json.loads((ROOT / f"results/v3/validation/bc_{variant}80_s{seed}_tuning.json").read_text())["rollouts"]
            assert all(a["initial_sensor_sha256"] == b["initial_sensor_sha256"] for a, b in zip(reference, other))
        identity.append(dict(seed=seed, shared_scene_sensors_exact=True))
    result = dict(results=results, scene_identity_checks=identity,
                  inference="Validation comparisons, paired seed/scene bootstrap (10,000 draws). Descriptive intervals, three seeds, no multiplicity correction; no final-test claim.")
    write(ROOT / "results/v3/ablation_summary.json", result)
    print(json.dumps(result), flush=True)
    return result
