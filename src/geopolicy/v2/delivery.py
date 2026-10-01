"""Read-only final evidence checks, including every saved rollout trace; no GPU use."""

import json
import math
import re
import zipfile
from pathlib import Path
from .config import file_hash, json_hash, source_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def verify_delivery():
    protocol = read("configs/v2/final_protocol.json")
    recipe = read("configs/v2/recipe.json")
    assert json_hash(recipe) == protocol["recipe_sha256"]
    for path, digest in protocol["evaluation_source_hashes"].items():
        assert source_hash(path) == digest, path
    for path, digest in protocol["checkpoints"].items():
        assert file_hash(path) == digest, path
    assert file_hash(recipe["dataset_manifest"]) == protocol["dataset_manifest_sha256"]
    assert file_hash("configs/v2/normalization.json") == protocol["normalization_sha256"]
    pretest = read("results/v2/pretest_verification.json")
    assert pretest["pytest"]["returncode"] == 0
    assert pretest["real_500_update_resume_bit_exact"]
    assert pretest["original_hdf5_hashes_verified"] == 274
    assert pretest["augmented_prefixes_array_exact"] == 221
    summary = read("results/v2/summary.json")
    cfg = recipe["stability"]
    groups = [
        ("main", "test", "main_rollouts", 3150),
        ("before_after", "before_after", "before_after_rollouts", 300),
        ("counterfactual", "counterfactual", "counterfactual_rollouts", 360),
    ]
    verified_rows = {}
    verified_steps = 0
    for label, folder, raw_name, expected_count in groups:
        assert summary["rollout_counts"][label] == expected_count
        assert len(read(f"results/v2/raw/{raw_name}.json")) == expected_count
        count = 0
        for result in sorted(Path(f"artifacts/v2/{folder}").glob("*/rollouts.json")):
            records = read(result)
            identities = {(r["condition"], r["scene_seed"]) for r in records}
            assert len(identities) == len(records), str(result)
            for row in records:
                assert row["checkpoint_sha256"] in protocol["registered_checkpoint_hashes"]
                trace_path = (
                    result.parent / "traces" / f"{row['condition']}_{row['scene_seed']}.json"
                )
                trace = read(trace_path)
                assert len(trace) == row["steps"] and len(trace) > 0, str(trace_path)
                assert trace[0]["action"] == row["first_action"]
                lifted = False
                started = None
                longest = 0.0
                stable = False
                previous_time = 0.0
                for step in trace:
                    assert len(step["action"]) == 7
                    assert all(math.isfinite(v) and abs(v) <= 1 for v in step["action"])
                    timestamp = step["time_s"]
                    assert timestamp > previous_time
                    previous_time = timestamp
                    z = step["selected_xyz_m"][2]
                    lifted |= z > cfg["minimum_previous_lift_m"]
                    valid = (
                        step["full_xy_containment"]
                        and cfg["center_height_m"][0] < z < cfg["center_height_m"][1]
                        and lifted
                        and not step["finger_contact_selected"]
                        and step["finger_width_m"] >= cfg["minimum_finger_width_m"]
                        and step["selected_linear_speed_m_s"] <= cfg["maximum_linear_speed_m_s"]
                        and step["selected_angular_speed_rad_s"]
                        <= cfg["maximum_angular_speed_rad_s"]
                    )
                    assert bool(valid) == step["valid_stability_sample"], str(trace_path)
                    if valid:
                        if started is None:
                            started = timestamp
                        longest = max(longest, timestamp - started)
                        stable |= timestamp - started + 1e-9 >= cfg["minimum_seconds"]
                    else:
                        started = None
                    assert stable == step["stable_success"], str(trace_path)
                assert stable == row["stable_success"], str(trace_path)
                assert abs(longest - row["longest_stable_seconds"]) < 1e-9
                assert (row["failure_stage"] == "success") == stable
                verified_steps += len(trace)
            count += len(records)
        assert count == expected_count, (label, count)
        verified_rows[label] = count
    clean = read("results/v2/clean_reproduction.json")
    assert clean["demo"]["verified"]
    assert all(
        r["all_actions_and_selected_object_positions_exact"] for r in clean["demo"]["episodes"]
    )
    portable = read("results/v2/portable_demo_verification.json")
    assert portable["verified"] and portable["no_training_dataset_or_original_weights_present"]
    assert all(
        r["all_actions_and_selected_object_positions_exact"] for r in portable["replay"]["episodes"]
    )
    bundle = Path("artifacts/v2/demo")
    meta = read(bundle / "bundle.json")
    assert file_hash(bundle / "policy.pt") == meta["checkpoint_sha256"]
    with zipfile.ZipFile("artifacts/v2/demo_bundle.zip") as archive:
        for path in bundle.iterdir():
            assert archive.read(path.name) == path.read_bytes(), path.name
    links = 0
    for document in [
        "README.md",
        "docs/v2_report.md",
        "docs/v2_design.md",
        "docs/v2_reproduction.md",
    ]:
        text = Path(document).read_text(encoding="utf8")
        for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if "://" in destination or destination.startswith("#"):
                continue
            assert (Path(document).parent / destination.split("#")[0]).exists(), (
                document,
                destination,
            )
            links += 1
    from geopolicy.io import save_json
    from .preservation import preserve_v1

    report = dict(
        verified=True,
        rollout_counts=verified_rows,
        trace_steps_verified=verified_steps,
        strict_score_recomputed_from_recorded_signals=True,
        geometry_containment_uses_saved_evaluator_flag=True,
        all_frozen_sources_and_registered_checkpoints_unchanged=True,
        compact_bundle_members_byte_exact=True,
        clean_demo_all_actions_and_object_positions_exact=True,
        isolated_demo_without_training_data_or_original_weights=True,
        local_document_links_verified=links,
        v1=preserve_v1(),
    )
    save_json("results/v2/delivery_verification.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify_delivery()
