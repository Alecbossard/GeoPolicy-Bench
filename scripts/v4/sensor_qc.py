"""Compare every saved test initial observation; NumPy only, no rollout changes.

Clean modalities use fixed_clean/s0/nominal for the same scene. Corrupted
preselection points/masks use fixed_clean/s0 for the same scene AND condition.
Raw RGB is provenance, not a direct PointCondition input. No outcome is read to
choose a reference, accept a difference, or remove an observation.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

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
SEEDS = (0, 1, 2)
SCENES = tuple(range(500000, 500020))
CAMERAS = ("agentview", "robot0_eye_in_hand")
CLEAN_FIELDS = ("rgb", "depth_m", "intrinsic", "base_from_camera", "points", "mask")
CLEAN_KEYS = ("state", "world_from_base") + tuple(
    f"{camera}__clean__{field}" for camera in CAMERAS for field in CLEAN_FIELDS
)
CORRUPTED_KEYS = tuple(
    f"{camera}__corrupted__{field}"
    for camera in CAMERAS
    for field in ("points", "mask")
)
EFFECTIVE_CLEAN_KEYS = tuple(k for k in CLEAN_KEYS if not k.endswith("__rgb"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def packet_digest(arrays: dict, mode: str, fields: tuple[str, ...]) -> str:
    """Reproduce the frozen evaluator digest without importing its runtime."""
    digest = hashlib.sha256()
    for camera in CAMERAS:
        for field in fields:
            array = np.ascontiguousarray(arrays[f"{camera}__{mode}__{field}"])
            for value in (camera, field, str(array.dtype), str(array.shape)):
                digest.update(value.encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(CLEAN_KEYS + CORRUPTED_KEYS):
            raise ValueError(f"Unexpected initial NPZ fields: {relative(path)}")
        arrays = {key: saved[key] for key in saved.files}
    for key, array in arrays.items():
        field = key.split("__")[-1]
        expected = {
            "state": ((23,), "float32"),
            "world_from_base": ((4, 4), "float32"),
            "rgb": ((128, 128, 3), "uint8"),
            "depth_m": ((128, 128), "float32"),
            "intrinsic": ((3, 3), "float32"),
            "base_from_camera": ((4, 4), "float32"),
            "points": ((512, 6), "float32"),
            "mask": ((512,), "bool"),
        }[field]
        if (array.shape, str(array.dtype)) != expected or not np.isfinite(array).all():
            raise ValueError(f"Invalid finite shape/dtype {key}: {relative(path)}")
    return arrays


def difference(observed: np.ndarray, reference: np.ndarray) -> dict | None:
    """Require byte equality, including shape and dtype; summarize changed values."""
    observed_bytes = np.ascontiguousarray(observed).tobytes()
    reference_bytes = np.ascontiguousarray(reference).tobytes()
    if (
        observed.shape == reference.shape
        and observed.dtype == reference.dtype
        and observed_bytes == reference_bytes
    ):
        return None
    if observed.shape != reference.shape or observed.dtype != reference.dtype:
        raise ValueError("Shapes/dtypes changed between validated initial observations")
    changed = np.any(
        np.frombuffer(observed_bytes, dtype=np.uint8).reshape(-1, observed.itemsize)
        != np.frombuffer(reference_bytes, dtype=np.uint8).reshape(
            -1, reference.itemsize
        ),
        axis=1,
    ).reshape(observed.shape)
    indices = np.argwhere(changed)
    shown = indices[:1024]
    return {
        "dtype": str(observed.dtype),
        "shape": list(observed.shape),
        "count": int(len(indices)),
        "indices": shown.tolist(),
        "indices_complete": len(shown) == len(indices),
        "reference": [reference[tuple(index)].item() for index in shown],
        "observed": [observed[tuple(index)].item() for index in shown],
        "max_absolute_difference": float(
            np.abs(observed.astype(np.float64) - reference.astype(np.float64)).max()
        ),
        "reference_array_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "observed_array_sha256": hashlib.sha256(observed_bytes).hexdigest(),
    }


def main() -> None:
    protocol_path = ROOT / "configs/v4/final_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    registry = {record["checkpoint"]: record for record in protocol["registry"]}
    indexed, result_shas = {}, {}
    paths = sorted((ROOT / "results/v4/test").glob("*.json"))
    if len(paths) != len(GROUPS) * len(SEEDS) * len(CONDITIONS):
        raise ValueError(f"Expected 120 raw test JSON files, found {len(paths)}")
    for path in paths:
        result = json.loads(path.read_text(encoding="utf8"))
        identity = result["identity"]
        record = registry[identity["checkpoint"]]
        if (
            identity["split"] != "test"
            or identity["first"] != SCENES[0]
            or identity["episodes"] != len(SCENES)
            or identity["checkpoint_sha256"] != record["sha256"]
            or identity["sources"] != protocol["sources"]
            or identity["plan_sha256"] != protocol["plan_sha256"]
            or len(result["rollouts"]) != len(SCENES)
        ):
            raise ValueError(
                f"Raw result identity/completeness differs: {relative(path)}"
            )
        result_shas[relative(path)] = sha(path)
        for row in result["rollouts"]:
            key = (
                row["group"],
                row["condition"],
                row["training_seed"],
                row["scene_seed"],
            )
            if (
                key in indexed
                or row["group"] != record["group"]
                or row["training_seed"] != record["training_seed"]
                or row["condition"] != identity["condition"]
                or row["checkpoint_sha256"] != record["sha256"]
            ):
                raise ValueError(f"Duplicate or inconsistent row identity: {key}")
            npz = (
                ROOT
                / "artifacts/v4/evaluations"
                / path.stem
                / "initial"
                / f"{key[3]}.npz"
            )
            indexed[key] = (row, npz, path)
    expected = {
        (group, condition, seed, scene)
        for group in GROUPS
        for condition in CONDITIONS
        for seed in SEEDS
        for scene in SCENES
    }
    if set(indexed) != expected:
        raise ValueError(
            f"Incomplete matrix: {len(expected - indexed.keys())} missing rows"
        )

    raw_mismatches, effective_mismatches, references = [], [], {}
    npz_shas, field_counts = {}, collections.Counter()
    for scene in SCENES:
        clean_reference = indexed[("fixed_clean", "nominal", 0, scene)][1]
        clean_arrays = load_arrays(clean_reference)
        condition_refs = {
            condition: indexed[("fixed_clean", condition, 0, scene)][1]
            for condition in CONDITIONS
        }
        corrupted_arrays = {
            condition: load_arrays(path) for condition, path in condition_refs.items()
        }
        references[str(scene)] = {
            "clean": {
                "path": relative(clean_reference),
                "sha256": sha(clean_reference),
            },
            "corrupted_by_condition": {
                condition: {"path": relative(path), "sha256": sha(path)}
                for condition, path in condition_refs.items()
            },
        }
        for group in GROUPS:
            for condition in CONDITIONS:
                for seed in SEEDS:
                    row, npz, result_path = indexed[(group, condition, seed, scene)]
                    npz_hash = sha(npz)
                    if npz_hash != row["initial_npz_sha256"]:
                        raise ValueError(
                            f"Saved NPZ SHA differs from raw row: {relative(npz)}"
                        )
                    npz_shas[relative(npz)] = npz_hash
                    arrays = load_arrays(npz)
                    clean_hash = hashlib.sha256(
                        (
                            packet_digest(arrays, "clean", CLEAN_FIELDS)
                            + hashlib.sha256(arrays["state"].tobytes()).hexdigest()
                        ).encode()
                    ).hexdigest()
                    corrupted_hash = packet_digest(
                        arrays, "corrupted", ("points", "mask")
                    )
                    if (
                        clean_hash != row["initial_clean_hash"]
                        or corrupted_hash != row["initial_corrupted_hash"]
                    ):
                        raise ValueError(
                            f"Saved sensor digest differs from raw row: {relative(npz)}"
                        )
                    for key in CLEAN_KEYS + CORRUPTED_KEYS:
                        corrupted = key in CORRUPTED_KEYS
                        ref_arrays = (
                            corrupted_arrays[condition] if corrupted else clean_arrays
                        )
                        ref_path = (
                            condition_refs[condition] if corrupted else clean_reference
                        )
                        field_counts[key] += 1
                        diff = difference(arrays[key], ref_arrays[key])
                        if diff is None:
                            continue
                        entry = dict(
                            scene_seed=scene,
                            group=group,
                            training_seed=seed,
                            condition=condition,
                            key=key,
                            raw_result=relative(result_path),
                            raw_result_sha256=result_shas[relative(result_path)],
                            observed_npz=relative(npz),
                            observed_npz_sha256=npz_hash,
                            reference_npz=relative(ref_path),
                            reference_npz_sha256=sha(ref_path),
                            **diff,
                        )
                        if not corrupted:
                            raw_mismatches.append(entry)
                        if corrupted or key in EFFECTIVE_CLEAN_KEYS:
                            effective_mismatches.append(entry)
    # Prove the QC did not change any raw result or saved initial observation.
    for relative_path, expected_sha in {**result_shas, **npz_shas}.items():
        if sha(ROOT / relative_path) != expected_sha:
            raise ValueError(f"Raw input changed during sensor QC: {relative_path}")
    report = {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checked_rollouts": len(indexed),
        "raw_result_sha256": result_shas,
        "initial_npz_sha256": npz_shas,
        "protocol_sha256": sha(protocol_path),
        "sensor_qc_helper_sha256": sha(Path(__file__)),
        "checked_raw_result_files": len(result_shas),
        "checked_initial_npz_files": len(npz_shas),
        "unique_scenes": len(SCENES),
        "corrupted_scene_condition_pairs": len(SCENES) * len(CONDITIONS),
        "all_initial_npz_sha256_match_rows": True,
        "all_clean_and_corrupted_digests_recomputed_match_rows": True,
        "all_raw_inputs_unchanged_during_qc": True,
        "all_effective_policy_initial_pairs_exact": not effective_mismatches,
        "all_raw_initial_modalities_exact": not raw_mismatches,
        "effective_mismatches": effective_mismatches,
        "raw_modality_mismatches": raw_mismatches,
        "checked_fields_per_key": dict(field_counts),
        "references_by_scene": references,
        "clean_reference_rule": "fixed_clean, training_seed=0, nominal, same scene",
        "corrupted_reference_rule": "fixed_clean, training_seed=0, same condition and scene",
        "effective_clean_fields": list(EFFECTIVE_CLEAN_KEYS),
        "effective_corrupted_fields": list(CORRUPTED_KEYS),
        "comparison": "Exact shape, dtype and contiguous bytes; no floating tolerance.",
        "input_contract": (
            "Both cameras' clean/corrupted XYZRGB points and masks before view selection "
            "and fusion are compared, plus state, depth, calibrations and world_from_base. "
            "Raw RGB images are retained provenance, not direct policy inputs. "
            "The same field rule applies to every group, seed, condition and outcome."
        ),
        "scope_limit": (
            "Initial observations only; this does not assert that later policy-dependent "
            "observations or trajectories are identical. The cause of raw differences "
            "has not been demonstrated. No rollout is filtered, replaced or rerun."
        ),
        "excluded_or_modified_rollouts": 0,
        "cause_of_raw_difference_demonstrated": False,
    }
    destination = ROOT / "results/v4/initial_sensor_qc.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf8",
    )
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "checked_rollouts",
                    "all_effective_policy_initial_pairs_exact",
                    "all_raw_initial_modalities_exact",
                )
            }
            | {
                "raw_field_differences": len(raw_mismatches),
                "effective_field_differences": len(effective_mismatches),
            }
        )
    )
    if effective_mismatches:
        raise ValueError("Effective initial sensor pairs differ; QC diagnostic saved")


if __name__ == "__main__":
    main()
