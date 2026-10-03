"""Independent V4 trace geometry/dwell and initial sensor corruption audit."""

import hashlib
import json
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from geopolicy.v4.common import read, write, sha
from geopolicy.v4.perturbations import CAMERAS, perturb, packet_digest, evaluation_rng


def main():
    cfg = read("configs/v3/plan.json")["stability"]
    p = read("configs/v4/plan.json")
    checked = 0
    steps = 0
    initial_checks = 0
    identity = {}
    coverage = {}
    sources = {}
    mismatches = []
    for path in sorted((ROOT / "results/v4/validation").glob("*.json")) + sorted(
        (ROOT / "results/v4/test").glob("*.json")
    ):
        saved = read(path)
        if "rollouts" not in saved:
            continue
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        out = ROOT / "artifacts/v4/evaluations" / path.stem
        assert (
            sha(ROOT / saved["identity"]["checkpoint"])
            == saved["identity"]["checkpoint_sha256"]
        )
        for row in saved["rollouts"]:
            trace = read(out / "traces" / f"{row['scene_seed']}.json")
            assert len(trace) == row["steps"]
            starts = {"physical": None, "strict": None}
            success = {"physical": False, "strict": False}
            released = False
            maximum_height = 0
            previous = None
            for step in trace:
                t = step["time_s"]
                if previous is not None:
                    assert abs(t - previous - 0.05) < 1e-8
                previous = t
                position = np.asarray(step["selected_xyz_m"])
                goal = np.asarray(step["goal_xyz_m"])
                rotation = np.asarray(step["rotation"])
                maximum_height = max(maximum_height, float(position[2]))
                assert step["previously_lifted"] == (
                    maximum_height > cfg["minimum_previous_lift_m"]
                )
                assert (
                    len(step["action"]) == 7
                    and np.isfinite(step["action"]).all()
                    and max(abs(v) for v in step["action"]) <= 1.000001
                )
                support = cfg["cube_half_extent_m"] * abs(rotation[:2]).sum(1)
                inside = bool(
                    np.all(
                        abs(position[:2] - goal[:2]) + support
                        <= np.asarray(cfg["tray_inner_half_xy_m"])
                        - cfg["containment_margin_m"]
                    )
                )
                assert inside == step["inside"]
                height = (
                    cfg["center_height_m"][0] < position[2] < cfg["center_height_m"][1]
                )
                geometry = (
                    inside
                    and height
                    and step["previously_lifted"]
                    and not step["finger_contact"]
                )
                opened = step["finger_width_m"] >= cfg["minimum_finger_width_m"]
                released |= bool(geometry and opened)
                assert released == step["release_seen"]
                slow = (
                    step["linear_speed_m_s"] <= cfg["maximum_linear_speed_m_s"]
                    and step["angular_speed_rad_s"]
                    <= cfg["maximum_angular_speed_rad_s"]
                )
                valid = dict(
                    physical=bool(
                        released
                        and inside
                        and height
                        and not step["finger_contact"]
                        and slow
                    ),
                    strict=bool(geometry and opened and slow),
                )
                for metric in valid:
                    assert valid[metric] == step[metric + "_valid"]
                    if not valid[metric]:
                        starts[metric] = None
                    elif starts[metric] is None:
                        starts[metric] = t
                    if (
                        starts[metric] is not None
                        and t - starts[metric] + 1e-9 >= cfg["minimum_seconds"]
                    ):
                        success[metric] = True
                    assert (
                        success[metric]
                        == step[
                            (
                                "physical_success"
                                if metric == "physical"
                                else "strict_v2_success"
                            )
                        ]
                    )
                steps += 1
            assert (
                success["physical"] == row["physical_success"]
                and success["strict"] == row["strict_v2_success"]
            )
            assert row["posture_only_failure"] == (
                success["physical"] and not success["strict"]
            )
            scene = row["scene_seed"]
            initial_path = out / "initial" / f"{scene}.npz"
            assert sha(initial_path) == row["initial_npz_sha256"]
            with np.load(initial_path) as f:
                packet = {
                    c: {
                        k: f[c + "__clean__" + k]
                        for k in (
                            "rgb",
                            "depth_m",
                            "intrinsic",
                            "base_from_camera",
                            "points",
                            "mask",
                        )
                    }
                    for c in CAMERAS
                }
                corrupted = perturb(
                    packet, p["conditions"][row["condition"]], evaluation_rng(scene, 0)
                )
                for c in CAMERAS:
                    for k in ("points", "mask"):
                        assert np.array_equal(
                            corrupted[c][k], f[c + "__corrupted__" + k]
                        )
                clean_hash = hashlib.sha256(
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
                        + hashlib.sha256(f["state"].tobytes()).hexdigest()
                    ).encode()
                ).hexdigest()
                assert (
                    clean_hash == row["initial_clean_hash"]
                    and packet_digest(corrupted) == row["initial_corrupted_hash"]
                )
                for family in ("nominal", "depth", "missing", "occlusion"):
                    nominal = perturb(
                        packet,
                        dict(family=family, intensity=0),
                        evaluation_rng(scene, 0),
                    )
                    assert packet_digest(nominal) == packet_digest(packet)
                lo = perturb(
                    packet, p["conditions"]["occlusion25"], evaluation_rng(scene, 0)
                )
                hi = perturb(
                    packet, p["conditions"]["occlusion60"], evaluation_rng(scene, 0)
                )
                assert not (hi[CAMERAS[0]]["mask"] & ~lo[CAMERAS[0]]["mask"]).any()
                assert np.array_equal(
                    hi[CAMERAS[1]]["points"], packet[CAMERAS[1]]["points"]
                )
            cov = read(out / "coverage" / f"{scene}.json")
            for item in cov:
                if row["condition"] == "absent":
                    assert item["corrupted"][CAMERAS[0]] == 0
                for c in CAMERAS:
                    assert item["corrupted"][c] <= item["clean"][c]
                    key = (row["group"], row["condition"], c)
                    s = coverage.setdefault(key, [0, 0])
                    s[0] += item["corrupted"][c]
                    s[1] += item["clean"][c]
            key = (saved["identity"]["split"], scene)
            if key in identity and identity[key] != row["initial_clean_hash"]:
                mismatches.append(
                    dict(
                        file=path.name,
                        scene=scene,
                        reference=identity[key],
                        observed=row["initial_clean_hash"],
                    )
                )
            else:
                identity[key] = row["initial_clean_hash"]
            checked += 1
            initial_checks += 1
    output = dict(
        checked_rollouts=checked,
        checked_steps=steps,
        initial_npz_recomputed=initial_checks,
        corruptions_recomputed_exact=True,
        zero_intensity_nominal_exact=True,
        occlusion_masks_nested=True,
        independent_geometry_and_dwell=True,
        initial_clean_mismatches=mismatches,
        observed_point_retention=[
            dict(
                group=g,
                condition=c,
                camera=v,
                retained=s[0],
                clean=s[1],
                ratio=s[0] / s[1] if s[1] else 0,
            )
            for (g, c, v), s in sorted(coverage.items())
        ],
        input_sources_sha256=sources,
    )
    write(ROOT / "results/v4/trace_sensor_audit.json", output)
    print(
        json.dumps(
            {
                k: v
                for k, v in output.items()
                if k not in ("input_sources_sha256", "observed_point_retention")
            }
        )
    )


if __name__ == "__main__":
    main()
