import numpy as np
from geopolicy.environment import CAMERAS
from geopolicy.v2.data import history_state, point_batch_from_packet
from geopolicy.evaluation import perturb


def test_causal_history_has_no_future_and_pads_initial_state():
    states = np.arange(4 * 23, dtype=np.float32).reshape(4, 23)
    norm = {"state_mean": np.zeros(23), "state_std": np.ones(23)}
    np.testing.assert_array_equal(history_state(states, 0, 3, norm), np.tile(states[0], 3))
    np.testing.assert_array_equal(history_state(states, 2, 3, norm), states[[2, 1, 0]].reshape(-1))


def test_missing_fixed_view_does_not_delete_wrist_only_points():
    p = np.zeros((512, 6), np.float32)
    p[:, 2] = 0.85
    p[:, 3] = 1
    packet = {"world_from_base": np.eye(4)}
    for c in CAMERAS:
        packet[c] = {
            "rgb": np.full((2, 2, 3), 255, np.uint8),
            "depth_m": np.ones((2, 2), np.float32),
            "intrinsic": np.eye(3),
            "base_from_camera": np.eye(4),
            "points": p.copy(),
            "mask": np.ones(512, bool),
        }
    cfg = {"view": "wrist", "point_budget": 512}
    before = point_batch_from_packet(packet, cfg)
    after = point_batch_from_packet(
        perturb(packet, "fixed_camera_missing", np.random.default_rng(0)), cfg
    )
    for x, y in zip(before, after):
        np.testing.assert_array_equal(x, y)
    fixed = point_batch_from_packet(packet, dict(cfg, view="fixed"))
    assert not fixed[1].any()
