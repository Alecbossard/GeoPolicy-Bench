"""Training-only heuristic: nearly unchanged sensors with abruptly changed labels."""

import json
from pathlib import Path
import h5py
import numpy as np

manifest = json.loads(Path("configs/dataset_manifest.json").read_text())
ids = set(manifest["selected_train_ids"])
events = []
candidates = 0
for row in manifest["episodes"]:
    if row["episode_id"] not in ids:
        continue
    with h5py.File(row["path"]) as f:
        state = f["state"][:]
        action = f["action"][:]
        delta = np.abs(np.diff(state, axis=0))
        near = (
            (delta[:, :7].max(1) < 0.0005)
            & (delta[:, 7:14].max(1) < 0.001)
            & (delta[:, 14:17].max(1) < 0.0005)
            & (delta[:, 17:21].max(1) < 0.001)
            & (delta[:, 21:].max(1) < 0.0001)
        )
        jumps = np.linalg.norm(np.diff(action[:, :3], axis=0), axis=1) > 0.3
        for t in np.flatnonzero(near & jumps):
            candidates += 1
            rgb_changes = []
            depth_changes = []
            for camera in ["agentview", "robot0_eye_in_hand"]:
                rgb = f[camera + "/rgb"][t : t + 2].astype(np.float32)
                depth = f[camera + "/depth_m"][t : t + 2]
                rgb_changes.append(float(np.abs(rgb[1] - rgb[0]).mean()))
                depth_changes.append(float(np.abs(depth[1] - depth[0]).mean()))
            if max(rgb_changes) < 0.15 and max(depth_changes) < 0.0001:
                events.append(
                    {
                        "episode_id": row["episode_id"],
                        "frame": int(t),
                        "instruction": row["instruction"],
                        "rgb_mean_abs_difference_uint8": rgb_changes,
                        "depth_mean_abs_difference_m": depth_changes,
                        "action_xyz_l2_change": float(
                            np.linalg.norm(action[t + 1, :3] - action[t, :3])
                        ),
                        "eef_max_abs_change_m": float(delta[t, 14:17].max()),
                    }
                )
report = {
    "training_episodes": len(ids),
    "robot_near_static_action_jump_candidates": candidates,
    "sensor_near_static_action_jumps": len(events),
    "examples": events[:30],
    "criteria": {
        "joint_delta_rad": 0.0005,
        "joint_velocity_delta_rad_s": 0.001,
        "eef_delta_m": 0.0005,
        "quaternion_delta": 0.001,
        "gripper_delta_m": 0.0001,
        "rgb_mean_abs_delta_uint8": 0.15,
        "depth_mean_abs_delta_m": 0.0001,
        "action_xyz_l2_jump": 0.3,
    },
    "interpretation": "Heuristic evidence of action-label ambiguity near timed teacher phase boundaries. Current-frame observations have no history/timer. Does not prove unlearnability or establish the cause of final policy failures.",
    "test_data_used": False,
}
Path("results/observation_aliasing_diagnostic.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
