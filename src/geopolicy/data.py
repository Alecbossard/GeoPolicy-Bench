"""Lossless episode storage, scene splits, and student-only batches."""

import hashlib
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
from .environment import SelectPlace, CAMERAS
from .sensors import camera_packet, student_state, fuse_points


def scene_split(seed):
    if 0 <= seed < 100000:
        return "train"
    if 100000 <= seed < 200000:
        return "validation"
    if seed >= 200000:
        return "test"
    raise ValueError("Negative scene seed")


def collect(out, episodes=1, first_seed=0, teacher=None, resolution=128):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = None
    if teacher:
        from stable_baselines3 import PPO

        model = PPO.load(teacher, device="cpu")
    provenance = "ppo_bc_initialized" if model else "scripted_reference"
    checkpoint_sha = hashlib.sha256(Path(teacher).read_bytes()).hexdigest() if teacher else None
    env = SelectPlace(cameras=True, resolution=resolution)
    records = []
    for existing in sorted(out.glob("episode_*.h5")):
        with h5py.File(existing) as f:
            record = json.loads(f.attrs["metadata"])
            record["bytes"] = existing.stat().st_size
            records.append(record)
    for seed in range(first_seed, first_seed + episodes):
        path = out / f"episode_{seed:06d}.h5"
        if path.exists():
            continue
        obs = env.reset_scene(seed)
        frames = []
        start = time.perf_counter()
        for step in range(env.horizon):
            if model:
                action, _ = model.predict(env.teacher_state(), deterministic=True)
            else:
                action = env.reference_action()
            packet = camera_packet(env, obs)
            packet["state"] = student_state(obs)
            packet["action"] = action.astype(np.float32)
            packet["instruction_tokens"] = np.r_[
                np.eye(2)[env.target_object], np.eye(2)[env.target_goal]
            ].astype(np.float32)
            frames.append(packet)
            obs, _, done, info = env.step(action)
            if info["success"] or done:
                break
        # Store final sensor frame to support re-reading transition endpoints.
        final = camera_packet(env, obs)
        metadata = {
            "scene_seed": seed,
            "episode_id": seed,
            "split": scene_split(seed),
            "instruction": env.instruction,
            "object_id": env.target_object,
            "goal_id": env.target_goal,
            "provenance": provenance,
            "teacher_checkpoint": teacher,
            "teacher_sha256": checkpoint_sha,
            "control_hz": 20,
            "scene_config": "select_place_v1_permuted_colors_jittered_objects_trays",
            "resolution": resolution,
            "frames": len(frames),
            "seconds": time.perf_counter() - start,
            **info,
        }
        tmp = path.with_suffix(".tmp")
        with h5py.File(tmp, "w") as f:
            f.attrs["metadata"] = json.dumps(metadata)
            for key in ["state", "action", "instruction_tokens", "timestamp_s", "world_from_base"]:
                f.create_dataset(
                    key,
                    data=np.array([frame[key] for frame in frames]),
                    compression="gzip",
                    shuffle=True,
                )
            for camera in CAMERAS:
                group = f.create_group(camera)
                for key in frames[0][camera]:
                    group.create_dataset(
                        key,
                        data=np.array([frame[camera][key] for frame in frames]),
                        compression="gzip",
                        shuffle=True,
                    )
                group.create_dataset("final_rgb", data=final[camera]["rgb"], compression="gzip")
                group.create_dataset(
                    "final_depth_m", data=final[camera]["depth_m"], compression="gzip"
                )
        os.replace(tmp, path)
        metadata["bytes"] = path.stat().st_size
        records.append(metadata)
        (out / "manifest.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(metadata), flush=True)
    env.close()
    return records


class Episodes:
    """Small cached offline batches. Ground-truth attrs never returned to model."""

    def __init__(self, root, split="train", mode="fusion", horizon=8, limit=None, view_dropout=0):
        self.mode = mode
        self.horizon = horizon
        self.episodes = []
        self.view_dropout = view_dropout
        self.dropout_rng = np.random.default_rng(0)
        if not 0 <= view_dropout <= 1:
            raise ValueError("Dropout probability must be in [0,1]")
        if view_dropout and mode != "fusion":
            raise ValueError("View dropout is a fusion training ablation")
        self.index = []
        for path in sorted(Path(root).glob("*.h5")):
            with h5py.File(path) as f:
                meta = json.loads(f.attrs["metadata"])
                if meta["split"] != split or not meta["success"]:
                    continue
                item = {key: f[key][:] for key in ["state", "action", "instruction_tokens"]}
                if mode == "act":
                    # Same recorded views; reduced input resolution declared for ACT.
                    item["rgb"] = np.stack([f[c]["rgb"][:, ::2, ::2, :] for c in CAMERAS], 1)
                else:
                    point_sets, masks = [], []
                    # Reading each whole array once is numerically identical to
                    # frame indexing but avoids thousands of HDF5 reopen/decode calls.
                    cameras = CAMERAS if mode == "fusion" else CAMERAS[:1]
                    cached = {c: (f[c]["points"][:], f[c]["mask"][:]) for c in cameras}
                    single_points = []
                    single_masks = []
                    for i in range(len(item["state"])):
                        views = [(cached[c][0][i], cached[c][1][i]) for c in cameras]
                        p, m = fuse_points(views)
                        point_sets.append(p)
                        masks.append(m)
                        if view_dropout:
                            singles = [fuse_points([v]) for v in views]
                            single_points.append(np.stack([s[0] for s in singles]))
                            single_masks.append(np.stack([s[1] for s in singles]))
                    item["points"] = np.array(point_sets)
                    item["mask"] = np.array(masks)
                    if view_dropout:
                        item["single_points"] = np.array(single_points)
                        item["single_masks"] = np.array(single_masks)
                eid = len(self.episodes)
                self.episodes.append(item)
                self.index.extend((eid, i) for i in range(len(item["state"])))
                if limit and len(self.episodes) >= limit:
                    break
        if not self.episodes:
            raise ValueError(f"No successful {split} episodes in {root}")

    def normalization(self):
        out = {}
        for key in ["state", "action"]:
            arr = np.concatenate([e[key] for e in self.episodes])
            out[key + "_mean"] = arr.mean(0).astype(np.float32)
            out[key + "_std"] = np.maximum(arr.std(0), 0.01).astype(np.float32)
        return out

    def batch(self, rng, batch_size, normalization):
        import torch

        output = {}
        samples = []
        for ix in rng.integers(len(self.index), size=batch_size):
            eid, t = self.index[ix]
            e = self.episodes[eid]
            future = np.minimum(t + np.arange(self.horizon), len(e["action"]) - 1)
            sample = {
                "state": (e["state"][t] - normalization["state_mean"]) / normalization["state_std"],
                "instruction": e["instruction_tokens"][t],
                "action": (e["action"][future] - normalization["action_mean"])
                / normalization["action_std"],
                "action_mask": t + np.arange(self.horizon) < len(e["action"]),
            }
            if self.mode == "act":
                sample["rgb"] = e["rgb"][t].transpose(0, 3, 1, 2).astype(np.float32) / 255
            else:
                if self.view_dropout and self.dropout_rng.random() < self.view_dropout:
                    view = self.dropout_rng.integers(2)
                    sample.update(
                        points=e["single_points"][t, view], point_mask=e["single_masks"][t, view]
                    )
                else:
                    sample.update(points=e["points"][t], point_mask=e["mask"][t])
            samples.append(sample)
        for key in samples[0]:
            output[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
        return output
