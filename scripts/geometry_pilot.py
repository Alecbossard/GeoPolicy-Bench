"""Ground truth used only for calibration QA, never policy input."""

import json
from pathlib import Path
import numpy as np
from geopolicy.environment import SelectPlace, CAMERAS
from geopolicy.sensors import camera_packet, deproject

env = SelectPlace(cameras=True)
obs = env.reset_scene(123)
packet = camera_packet(env, obs)
world_from_base = packet["world_from_base"]
rows = []
for camera in CAMERAS:
    view = packet[camera]
    points = deproject(view["rgb"], view["depth_m"], view["intrinsic"], view["base_from_camera"])
    world = points[:, :3] @ world_from_base[:3, :3].T + world_from_base[:3, 3]
    color = points[:, 3:]
    for i in range(2):
        colored = (
            (
                (color[:, 0] > 0.2)
                & (color[:, 0] > color[:, 1] * 1.8)
                & (color[:, 0] > color[:, 2] * 1.8)
            )
            if i == 0
            else (
                (color[:, 1] > 0.2)
                & (color[:, 1] > color[:, 0] * 1.8)
                & (color[:, 1] > color[:, 2] * 1.8)
            )
        )
        colored &= (world[:, 2] > 0.79) & (world[:, 2] < 0.88)
        count = int(colored.sum())
        if count:
            estimate = np.median(world[colored], axis=0)
            truth = env.object_positions()[i]
            xy_error = float(np.linalg.norm(estimate[:2] - truth[:2]))
            rows.append(
                dict(
                    camera=camera,
                    object_id=i,
                    visible_pixels=count,
                    estimated_world=estimate.tolist(),
                    truth=truth.tolist(),
                    xy_error_m=xy_error,
                )
            )
            assert xy_error < 0.025, (camera, i, estimate, truth)
    assert abs(packet["timestamp_s"] - env.sim.data.time) < 1e-10
initial_extrinsic = packet["robot0_eye_in_hand"]["base_from_camera"]
for _ in range(10):
    obs, _, _, _ = env.step(np.array([0.2, 0, 0, 0, 0, 0, -1]))
new = camera_packet(env, obs)["robot0_eye_in_hand"]["base_from_camera"]
assert np.linalg.norm(initial_extrinsic - new) > 0.01
report = {
    "checks": rows,
    "wrist_extrinsic_change_l2": float(np.linalg.norm(initial_extrinsic - new)),
    "timestamps_synchronized": True,
}
Path("artifacts/geometry_pilot.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
env.close()
