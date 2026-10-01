"""Shared recorded/live student inputs, named cameras and causal state history."""

import json
from pathlib import Path
import h5py
import numpy as np
import torch
from geopolicy.environment import CAMERAS
from geopolicy.evaluation import perturb, rgb_packet
from geopolicy.sensors import camera_packet, student_state, fuse_points


def view_names(view):
    return {"fixed": CAMERAS[:1], "wrist": CAMERAS[1:], "fusion": CAMERAS}[view]


def history_state(states, index, length, normalization):
    indices = np.maximum(index - np.arange(length), 0)
    return (
        ((states[indices] - normalization["state_mean"]) / normalization["state_std"])
        .reshape(-1)
        .astype(np.float32)
    )


class Dataset:
    def __init__(self, recipe, cfg, split="train"):
        self.cfg = cfg
        manifest = json.loads(Path(recipe["dataset_manifest"]).read_text())
        allowed = (
            manifest["selected_train_ids"]
            if split == "train"
            else manifest["selected_validation_ids"]
        )
        self.episodes = []
        self.index = []
        for record in manifest["episodes"]:
            if record["episode_id"] not in allowed:
                continue
            assert record["split"] == split and record["success"]
            with h5py.File(record["path"]) as f:
                item = {k: f[k][:] for k in ["state", "action", "instruction_tokens"]}
                item["episode_id"] = record["episode_id"]
                if cfg["mode"].startswith("act_rgb"):
                    stride = 128 // cfg.get("rgb_resolution", 64)
                    item["rgb"] = np.stack(
                        [f[c + "/rgb"][:, ::stride, ::stride] for c in CAMERAS], 1
                    )
                else:
                    cameras = view_names(cfg["view"])
                    cached = {c: (f[c + "/points"][:], f[c + "/mask"][:]) for c in cameras}
                    points, masks = zip(
                        *(
                            fuse_points(
                                [(cached[c][0][t], cached[c][1][t]) for c in cameras],
                                n=cfg["point_budget"],
                            )
                            for t in range(len(item["state"]))
                        )
                    )
                    item["points"], item["point_mask"] = np.stack(points), np.stack(masks)
            eid = len(self.episodes)
            self.episodes.append(item)
            self.index.extend((eid, t) for t in range(len(item["state"])))
        assert [
            e["episode_id"] for e in self.episodes
        ] == allowed, "Frozen demonstration order changed"

    def normalization(self):
        if self.cfg.get("normalization_path"):
            return {
                k: np.asarray(v, np.float32)
                for k, v in json.loads(Path(self.cfg["normalization_path"]).read_text()).items()
            }
        return {
            name
            + suffix: (
                array.mean(0) if suffix == "_mean" else np.maximum(array.std(0), 0.01)
            ).astype(np.float32)
            for name in ["state", "action"]
            for array in [np.concatenate([e[name] for e in self.episodes])]
            for suffix in ["_mean", "_std"]
        }

    def batch(self, rng, batch_size, norm):
        samples = []
        for ix in rng.integers(len(self.index), size=batch_size):
            eid, t = self.index[ix]
            e = self.episodes[eid]
            indices = np.minimum(t + np.arange(self.cfg["horizon"]), len(e["action"]) - 1)
            sample = {
                "state": history_state(e["state"], t, self.cfg["history"], norm),
                "instruction": e["instruction_tokens"][t],
                "action": (e["action"][indices] - norm["action_mean"]) / norm["action_std"],
                "action_mask": t + np.arange(self.cfg["horizon"]) < len(e["action"]),
            }
            if "rgb" in e:
                sample["rgb"] = e["rgb"][t].transpose(0, 3, 1, 2).astype(np.float32) / 255
            else:
                sample.update(points=e["points"][t], point_mask=e["point_mask"][t])
            samples.append(sample)
        return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}


def point_batch_from_packet(packet, cfg):
    return fuse_points(
        [(packet[c]["points"], packet[c]["mask"]) for c in view_names(cfg["view"])],
        n=cfg["point_budget"],
    )


def live_batch(env, obs, history, cfg, norm, condition="nominal", rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    states = np.stack(history[-cfg["history"] :])
    batch = {
        "state": history_state(states, len(states) - 1, cfg["history"], norm),
        "instruction": np.r_[np.eye(2)[env.target_object], np.eye(2)[env.target_goal]].astype(
            np.float32
        ),
    }
    if cfg["mode"].startswith("act_rgb"):
        packet = rgb_packet(obs, condition, rng)
        stride = 128 // cfg.get("rgb_resolution", 64)
        batch["rgb"] = (
            np.stack(
                [packet[c]["rgb"][::stride, ::stride].transpose(2, 0, 1) for c in CAMERAS]
            ).astype(np.float32)
            / 255
        )
    else:
        # Always perturb a packet with named fixed+wrist entries, then select views.
        # V1's positional `i == 0` perturbation must never reinterpret wrist as fixed.
        packet = perturb(camera_packet(env, obs), condition, rng)
        batch["points"], batch["point_mask"] = point_batch_from_packet(packet, cfg)
    return {k: torch.from_numpy(np.asarray(v)[None]) for k, v in batch.items()}
