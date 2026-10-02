"""Independent box support-function containment and dwell recomputation."""
import json
import numpy as np
from .common import ROOT, sha, write


def audit():
    cfg = json.loads((ROOT / "configs/v3/plan.json").read_text())["stability"]
    checked_steps, checked_rows = 0, 0
    summaries = {}
    for path in sorted((ROOT / "results/v3/validation").glob("*.json")):
        saved = json.loads(path.read_text())
        out = ROOT / "artifacts/v3/evaluations" / path.stem
        if not (out / "identity.json").exists():
            continue
        identity = saved["identity"]
        if identity["checkpoint_sha256"]:
            # Identify the corresponding immutable saved checkpoint by hash.
            matches = [p for p in (ROOT / "artifacts/v3/runs").glob("*/best.pt")
                       if sha(p) == identity["checkpoint_sha256"]]
            assert matches, f"No local matching checkpoint for {path.name}"
        for row in saved["rollouts"]:
            trace = json.loads((out / "traces" / f"{row['scene_seed']}.json").read_text())
            assert len(trace) == row["steps"]
            starts = dict(physical=None, strict=None)
            success = dict(physical=False, strict=False)
            released, previous = False, None
            for step in trace:
                timestamp = step["time_s"]
                if previous is not None:
                    assert abs(timestamp - previous - .05) < 1e-8
                previous = timestamp
                position = np.asarray(step["selected_xyz_m"])
                rotation = np.asarray(step["rotation"])
                goal = np.asarray(step["goal_xyz_m"])
                # Independent of the eight-corner implementation in the evaluator.
                support = cfg["cube_half_extent_m"] * np.abs(rotation[:2]).sum(1)
                inside = bool(np.all(np.abs(position[:2] - goal[:2]) + support <=
                                    np.asarray(cfg["tray_inner_half_xy_m"]) - cfg["containment_margin_m"]))
                assert inside == step["inside"]
                height = cfg["center_height_m"][0] < position[2] < cfg["center_height_m"][1]
                geometry = inside and height and step["previously_lifted"] and not step["finger_contact"]
                open_hand = step["finger_width_m"] >= cfg["minimum_finger_width_m"]
                released |= bool(geometry and open_hand)
                assert released == step["release_seen"]
                slow = (step["linear_speed_m_s"] <= cfg["maximum_linear_speed_m_s"] and
                        step["angular_speed_rad_s"] <= cfg["maximum_angular_speed_rad_s"])
                valid = dict(physical=bool(released and inside and height and not step["finger_contact"] and slow),
                             strict=bool(geometry and open_hand and slow))
                for metric in ("physical", "strict"):
                    assert valid[metric] == step[metric + "_valid"]
                    if not valid[metric]:
                        starts[metric] = None
                    elif starts[metric] is None:
                        starts[metric] = timestamp
                    if starts[metric] is not None and timestamp - starts[metric] + 1e-9 >= cfg["minimum_seconds"]:
                        success[metric] = True
                    recorded_key = "physical_success" if metric == "physical" else "strict_v2_success"
                    assert success[metric] == step[recorded_key]
                checked_steps += 1
            assert success["physical"] == row["physical_success"]
            assert success["strict"] == row["strict_v2_success"]
            assert row["posture_only_failure"] == (success["physical"] and not success["strict"])
            checked_rows += 1
        rows = saved["rollouts"]
        summaries[path.stem] = dict(episodes=len(rows), physical=sum(r["physical_success"] for r in rows),
                                   strict=sum(r["strict_v2_success"] for r in rows))
    result = dict(checked_rows=checked_rows, checked_steps=checked_steps,
                  independent_containment="box support function, reconstructed from saved pose and goal",
                  summaries=summaries)
    write(ROOT / "results/v3/trace_audit.json", result)
    print(json.dumps(result), flush=True)
    return result
