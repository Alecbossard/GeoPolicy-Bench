"""Metric RGB-D geometry; OpenCV camera frame -> robot base frame."""

import numpy as np


def deproject(rgb, depth, intrinsic, base_from_camera):
    depth = np.asarray(depth).squeeze()
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    valid = np.isfinite(depth) & (depth > 0) & (depth < 5)
    z = depth[valid]
    xyz = np.stack(
        [
            (u[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (v[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ],
        -1,
    )
    xyz = xyz @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]
    colors = np.asarray(rgb)[valid].astype(np.float32) / 255
    return np.c_[xyz, colors].astype(np.float32)


def sample_points(points, n=512, voxel=0.004):
    points = np.asarray(points, np.float32)
    if voxel and len(points):
        # Deterministic first point per voxel; preserves color and geometry.
        keys = np.floor(points[:, :3] / voxel).astype(np.int64)
        shifted = keys - keys.min(0)
        spans = shifted.max(0) + 1
        packed = (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2]
        _, indices = np.unique(packed, return_index=True)
        points = points[np.sort(indices)]
    out = np.zeros((n, 6), np.float32)
    mask = np.zeros(n, bool)
    if len(points):
        idx = np.linspace(0, len(points) - 1, min(n, len(points)), dtype=int)
        out[: len(idx)] = points[idx]
        mask[: len(idx)] = True
    return out, mask


def camera_packet(env, obs, voxel=0.004, camera_names=None):
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix,
        get_camera_extrinsic_matrix,
        get_real_depth_map,
    )
    from .environment import CAMERAS

    # Panda base fixed; camera pose (including wrist) is read for each timestamp.
    base_id = env.sim.model.body_name2id("robot0_base")
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = env.sim.data.body_xmat[base_id].reshape(3, 3)
    world_from_base[:3, 3] = env.sim.data.body_xpos[base_id]
    base_from_world = np.linalg.inv(world_from_base)
    packet = {}
    for name in CAMERAS if camera_names is None else camera_names:
        rgb = obs[name + "_image"][::-1].copy()
        depth = get_real_depth_map(env.sim, obs[name + "_depth"])[::-1].squeeze().astype(np.float32)
        k = get_camera_intrinsic_matrix(env.sim, name, *depth.shape)
        world_from_cam = get_camera_extrinsic_matrix(env.sim, name)
        base_from_cam = base_from_world @ world_from_cam
        points = deproject(rgb, depth, k, base_from_cam)
        # Fixed world workspace crop, never object-specific or segmentation-based.
        world = points[:, :3] @ world_from_base[:3, :3].T + world_from_base[:3, 3]
        crop = (
            (world[:, 0] > -0.25)
            & (world[:, 0] < 0.25)
            & (abs(world[:, 1]) < 0.31)
            & (world[:, 2] > 0.79)
            & (world[:, 2] < 1.15)
        )
        p, mask = sample_points(points[crop], voxel=voxel)
        packet[name] = {
            "rgb": rgb,
            "depth_m": depth,
            "intrinsic": k.astype(np.float32),
            "base_from_camera": base_from_cam.astype(np.float32),
            "points": p,
            "mask": mask,
        }
    packet["world_from_base"] = world_from_base.astype(np.float32)
    packet["timestamp_s"] = float(env.sim.data.time)
    return packet


def student_state(obs):
    """Strict allowlist: robot observations only, no teacher/object state."""
    keys = [
        "robot0_joint_pos",
        "robot0_joint_vel",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    return np.concatenate([obs[key] for key in keys]).astype(np.float32)


def fuse_points(views, n=512, voxel=0.004):
    valid = [p[m] for p, m in views]
    return sample_points(np.concatenate(valid, axis=0), n=n, voxel=voxel)
