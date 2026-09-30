import numpy as np
from geopolicy.sensors import deproject, sample_points, fuse_points, student_state


def test_camera_plane_and_rigid_transform():
    rgb = np.full((2, 2, 3), 255, np.uint8)
    depth = np.ones((2, 2), np.float32) * 2
    k = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    p = deproject(rgb, depth, k, t)
    np.testing.assert_allclose(p[:, :3], [[1, 2, 5], [2, 2, 5], [1, 3, 5], [2, 3, 5]])
    np.testing.assert_allclose(p[:, 3:], 1)


def test_invalid_depth_and_empty_camera_mask():
    p = deproject(np.zeros((2, 2, 3)), np.array([[0, np.nan], [np.inf, -1]]), np.eye(3), np.eye(4))
    sampled, mask = sample_points(p)
    assert not mask.any() and np.isfinite(sampled).all()
    fusion, fused_mask = fuse_points([(sampled, mask)])
    assert not fused_mask.any()


def test_duplicate_voxels_and_view_fusion():
    p = np.array([[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0.1, 0, 1, 0, 1, 0]], np.float32)
    a, mask = sample_points(p, 4)
    assert mask.sum() == 2
    b, bmask = fuse_points([(a, mask), (a, mask)], 4)
    assert bmask.sum() == 2


def test_packed_voxel_keys_preserve_original_exact_representatives():
    rng = np.random.default_rng(19)
    p = rng.uniform(-1, 1, (3000, 6)).astype(np.float32)
    p = np.concatenate([p, p[:100]], axis=0)
    _, old_indices = np.unique(
        np.floor(p[:, :3] / 0.02).astype(np.int64), axis=0, return_index=True
    )
    original = p[np.sort(old_indices)]
    old = original[np.linspace(0, len(original) - 1, 512, dtype=int)]
    new, mask = sample_points(p, 512, 0.02)
    np.testing.assert_array_equal(old, new)
    assert mask.all()


def test_privileged_state_cannot_enter_student():
    obs = {
        k: np.array([1.0])
        for k in [
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    }
    obs.update(
        cube_pos=np.array([999.0]), phase=np.array([999.0]), target_position=np.array([999.0])
    )
    assert student_state(obs).tolist() == [1.0] * 5
