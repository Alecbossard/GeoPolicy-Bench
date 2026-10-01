"""Measured trace features; correlations are not labels of a proven root cause."""

import json
from collections import Counter
from pathlib import Path
from geopolicy.io import save_json


def analyze_folder(folder):
    root = Path(folder)
    rows = json.loads((root / "rollouts.json").read_text())
    details = []
    for row in rows:
        trace = json.loads(
            (root / "traces" / f"{row['condition']}_{row['scene_seed']}.json").read_text()
        )
        signs = [
            1 if t["gripper_command"] > 0 else (-1 if t["gripper_command"] < 0 else 0)
            for t in trace
        ]
        switches = sum(a != b for a, b in zip(signs, signs[1:]))
        lift_seen = False
        candidates = []
        for i, t in enumerate(trace):
            lift_seen |= t["selected_xyz_m"][2] > 0.89
            if (
                lift_seen
                and t["full_xy_containment"]
                and 0.816 < t["selected_xyz_m"][2] < 0.855
                and not t["finger_contact_selected"]
                and t["finger_width_m"] >= 0.045
            ):
                candidates.append(i)
        first = min(candidates) if candidates else None
        tail = trace[first:] if first is not None else []
        details.append(
            dict(
                scene_seed=row["scene_seed"],
                condition=row["condition"],
                failure_stage=row["failure_stage"],
                stable_success=row["stable_success"],
                gripper_sign_switches=switches,
                first_contained_release_step=first,
                closing_command_after_candidate=bool(
                    tail and any(t["gripper_command"] > 0 for t in tail)
                ),
                contact_after_candidate=bool(
                    tail and any(t["finger_contact_selected"] for t in tail)
                ),
                left_xy_containment_after_candidate=bool(
                    tail and any(not t["full_xy_containment"] for t in tail)
                ),
                maximum_linear_speed_after_candidate=max(
                    (t["selected_linear_speed_m_s"] for t in tail), default=None
                ),
                maximum_angular_speed_after_candidate=max(
                    (t["selected_angular_speed_rad_s"] for t in tail), default=None
                ),
            )
        )
    return dict(
        episodes=len(rows),
        stage_counts=dict(Counter(r["failure_stage"] for r in rows)),
        episodes_with_contained_release=sum(
            d["first_contained_release_step"] is not None for d in details
        ),
        failures_with_contained_release=sum(
            d["first_contained_release_step"] is not None and not d["stable_success"]
            for d in details
        ),
        details=details,
        interpretation="Trace events describe behavior under continued policy control. A closing command/contact or loss of containment after candidate release is an association, not an isolated causal intervention.",
    )


def analyze_v1():
    result = {
        name: analyze_folder(f"artifacts/v2/diagnostics/{name}")
        for name in ["act_v1", "mono_v1", "fusion_v1", "reference"]
    }
    save_json("results/v2/v1_failure_traces.json", result)
    return result
