"""Closed-loop student-only evaluation and sensor perturbations."""

import json
import csv
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from .environment import SelectPlace, CAMERAS
from .sensors import camera_packet, student_state, deproject, sample_points, fuse_points
from .policies import make_policy
from .io import save_json

CONDITIONS = ("nominal", "occlusion", "depth_degraded", "fixed_camera_missing", "extrinsic_error")


def occlusion_slice(h, w, rng):
    local_rng = np.random.default_rng()
    local_rng.bit_generator.state = rng.bit_generator.state
    cy, cx = local_rng.uniform(0.5, 0.6), local_rng.uniform(0.43, 0.57)
    rh, rw = local_rng.uniform(0.3, 0.42), local_rng.uniform(0.25, 0.4)
    return slice(int((cy - rh / 2) * h), int((cy + rh / 2) * h)), slice(
        int((cx - rw / 2) * w), int((cx + rw / 2) * w)
    )


def rgb_packet(obs, condition, rng):
    packet = {c: {"rgb": obs[c + "_image"][::-1].copy()} for c in CAMERAS}
    if condition == "occlusion":
        image = packet[CAMERAS[0]]["rgb"]
        slices = occlusion_slice(*image.shape[:2], rng)
        image[slices] = 0
    elif condition == "fixed_camera_missing":
        packet[CAMERAS[0]]["rgb"][:] = 0
    # Depth/calibration perturbations do not alter RGB-only model inputs.
    return packet


def perturb(packet, condition, rng, voxel=0.004):
    # Same simulated scene/seed across all conditions, sensor-space corruption.
    for i, camera in enumerate(c for c in CAMERAS if c in packet):
        view = packet[camera]
        if condition == "occlusion" and i == 0:
            h, w = view["depth_m"].shape
            # Per-scene rectangle is static through the rollout. The RNG clone
            # avoids advancing the scene's perturbation generator for this mode.
            slices = occlusion_slice(h, w, rng)
            view["rgb"][slices] = 0
            view["depth_m"][slices] = np.nan
        if condition == "depth_degraded":
            depth = view["depth_m"]
            depth += rng.normal(0, 0.005, depth.shape).astype(np.float32)
            depth[rng.random(depth.shape) < 0.15] = np.nan
        if condition == "fixed_camera_missing" and i == 0:
            view["rgb"][:] = 0
            view["depth_m"][:] = np.nan
        if condition == "extrinsic_error" and i == 0:
            # Calibration mismatch only, not simulator camera movement.
            angle = np.deg2rad(5)
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            )
            t = view["base_from_camera"].copy()
            t[:3, :3] = rotation @ t[:3, :3]
            t[:3, 3] += [0.01, -0.01, 0]
            view["base_from_camera"] = t
        if condition == "depth_degraded" or (condition != "nominal" and i == 0):
            points = deproject(
                view["rgb"], view["depth_m"], view["intrinsic"], view["base_from_camera"]
            )
            world_from_base = packet["world_from_base"]
            world = points[:, :3] @ world_from_base[:3, :3].T + world_from_base[:3, 3]
            crop = (
                (abs(world[:, 0]) < 0.25)
                & (abs(world[:, 1]) < 0.31)
                & (world[:, 2] > 0.79)
                & (world[:, 2] < 1.15)
            )
            view["points"], view["mask"] = sample_points(points[crop], voxel=voxel)
    return packet


def observation_batch(env, obs, mode, norm, condition="nominal", rng=None, voxel=0.004):
    rng = rng if rng is not None else np.random.default_rng(0)
    cameras = CAMERAS if mode == "fusion" else CAMERAS[:1]
    packet = (
        rgb_packet(obs, condition, rng)
        if mode == "act"
        else perturb(
            camera_packet(env, obs, voxel=voxel, camera_names=cameras), condition, rng, voxel
        )
    )
    state = student_state(obs)
    batch = {
        "state": (state - norm["state_mean"]) / norm["state_std"],
        "instruction": np.r_[np.eye(2)[env.target_object], np.eye(2)[env.target_goal]].astype(
            np.float32
        ),
    }
    if mode == "act":
        batch["rgb"] = (
            np.stack([packet[c]["rgb"][::2, ::2].transpose(2, 0, 1) for c in CAMERAS]).astype(
                np.float32
            )
            / 255
        )
    else:
        views = [
            (packet[c]["points"], packet[c]["mask"])
            for c in (CAMERAS if mode == "fusion" else CAMERAS[:1])
        ]
        batch["points"], batch["point_mask"] = fuse_points(views, voxel=voxel)
    return {k: torch.from_numpy(np.asarray(v)[None]) for k, v in batch.items()}, packet


def evaluate_student(
    checkpoint,
    out,
    episodes=20,
    first_seed=100100,
    conditions=("nominal",),
    video_count=0,
    object_id=None,
    goal_id=None,
    device=None,
    execute_steps=None,
    onnx_path=None,
    voxel=0.004,
):
    checkpoint = Path(checkpoint)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    mode = config["mode"]
    norm = saved["normalization"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    execute_steps = execute_steps or config.get("execute_steps", 1)
    if first_seed >= 200000:
        protocol = json.loads(Path("configs/benchmark_protocol.json").read_text())
        if not protocol["frozen"]:
            raise ValueError("Final test requires a frozen protocol")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    identity = {
        "checkpoint_sha256": checkpoint_sha,
        "device": device,
        "execute_steps": execute_steps,
        "object_id": object_id,
        "goal_id": goal_id,
        "voxel_m": voxel,
        "onnx_sha256": (
            hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest() if onnx_path else None
        ),
    }
    identity_path = out / "evaluation_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(
            "Existing evaluation belongs to a different checkpoint/recipe; use a new output directory"
        )
    identity_path.write_text(json.dumps(identity, indent=2))
    torch.set_num_threads(2)
    model = make_policy(
        mode,
        prediction_type=config.get("prediction_type", "epsilon"),
        color_prior=config.get("color_prior", False),
    ).to(device)
    model.load_state_dict(
        saved["extra"]["ema"] if saved.get("extra") and "ema" in saved["extra"] else saved["model"]
    )
    model.eval()
    env = SelectPlace(cameras=True)
    if onnx_path:
        if device != "cpu" or mode == "act":
            raise ValueError("Validated ONNX backend is CPU diffusion denoiser")
        from .exporting import install_onnx_denoiser

        install_onnx_denoiser(model, onnx_path)
    rows = (
        json.loads((out / "rollouts.json").read_text()) if (out / "rollouts.json").exists() else []
    )
    completed = {(r["condition"], r["scene_seed"]) for r in rows}
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError(condition)
        for seed in range(first_seed, first_seed + episodes):
            if (condition, seed) in completed:
                continue
            obs = env.reset_scene(seed, object_id, goal_id)
            if object_id is not None or goal_id is not None:
                digest = hashlib.sha256(student_state(obs).tobytes())
                for camera in CAMERAS:
                    digest.update(obs[camera + "_image"].tobytes())
                    digest.update(obs[camera + "_depth"].tobytes())
                digest.update(env.sim.data.cam_xpos.tobytes())
                digest.update(env.sim.data.cam_xmat.tobytes())
                initial_sensor_sha = digest.hexdigest()
            torch.manual_seed(seed + config.get("seed", 0) * 1000000)
            rng = np.random.default_rng(seed)
            preprocessing = []
            policy_time = []
            frames = []
            start = time.perf_counter()
            action_queue = []
            for step in range(env.horizon):
                if not action_queue:
                    t = time.perf_counter()
                    batch, packet = observation_batch(env, obs, mode, norm, condition, rng, voxel)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if device == "cuda":
                        torch.cuda.synchronize()
                    preprocessing.append(time.perf_counter() - t)
                    t = time.perf_counter()
                    with torch.inference_mode():
                        normalized = model.predict(batch)[0, :execute_steps].cpu().numpy()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    policy_time.append(time.perf_counter() - t)
                    action_queue = list(
                        np.clip(normalized * norm["action_std"] + norm["action_mean"], -1, 1)
                    )
                action = action_queue.pop(0)
                if step == 0:
                    first_action = action.tolist()
                if seed - first_seed < video_count:
                    frames.append(
                        np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], axis=1)
                    )
                obs, _, done, info = env.step(action)
                if done or info["success"]:
                    break
            if frames:
                frames.append(np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], axis=1))
            record = {
                "scene_seed": seed,
                "training_seed": config.get("seed"),
                "mode": mode,
                "condition": condition,
                "checkpoint": str(checkpoint),
                "instruction": env.instruction,
                "object_id": env.target_object,
                "goal_id": env.target_goal,
                "first_action": first_action,
                "checkpoint_sha256": checkpoint_sha,
                "backend": "onnx_denoiser" if onnx_path else "pytorch",
                "voxel_m": voxel,
                "inference_device": device,
                "execute_steps": execute_steps,
                "steps": step + 1,
                "wall_seconds": time.perf_counter() - start,
                "simulation_seconds": (step + 1) / 20,
                "preprocess_p50_ms": float(np.percentile(preprocessing, 50) * 1000),
                "preprocess_p95_ms": float(np.percentile(preprocessing, 95) * 1000),
                "policy_p50_ms": float(np.percentile(policy_time, 50) * 1000),
                "policy_p95_ms": float(np.percentile(policy_time, 95) * 1000),
                "total_p50_ms": float(
                    np.percentile(np.array(preprocessing) + policy_time, 50) * 1000
                ),
                "total_p95_ms": float(
                    np.percentile(np.array(preprocessing) + policy_time, 95) * 1000
                ),
                **info,
            }
            if object_id is not None or goal_id is not None:
                record["initial_rgbd_robot_camera_pose_sha256"] = initial_sensor_sha
            rows.append(record)
            if frames:
                import imageio.v2 as iio

                iio.mimwrite(
                    out
                    / f"{mode}_s{config.get('seed')}_{condition}_{seed}_{'success' if info['success'] else 'failure'}.mp4",
                    frames,
                    fps=20,
                )
            save_json(out / "rollouts.json", rows)
            with (out / "rollouts.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(record), flush=True)
    env.close()
    return rows
