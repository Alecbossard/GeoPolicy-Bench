"""Same79 immutable V3 demonstrations/actions, augmentation RNG kept separate."""

import h5py
import numpy as np
import torch
from geopolicy.v2.data import history_state
from .common import ROOT, read, sha
from .perturbations import CAMERAS, perturb, fuse, augmentation_condition


class Data:
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.episodes = []
        self.index = []
        self.identities = []
        records = read(
            "configs/v3/datasets/" + cfg["dataset_manifest_sha256"] + ".json"
        )
        selected = [r for r in records if r["split"] == split and r["success"]]
        assert len(selected) == (79 if split == "train" else 10)
        for record in selected:
            path = ROOT / record["path"]
            assert sha(path) == record["sha256"]
            count = record["frames"]
            with h5py.File(path) as f:
                e = {k: f[k][:count] for k in ("state", "action", "instruction_tokens")}
                e["cameras"] = {
                    c: {
                        k: f[c][k][:count]
                        for k in ("points", "mask", "intrinsic", "base_from_camera")
                    }
                    for c in CAMERAS
                }
                # Cached nominal fusion reproduces V3 bit for bit.
                pairs = [
                    fuse(
                        {
                            c: {k: v[t] for k, v in e["cameras"][c].items()}
                            for c in CAMERAS
                        },
                        cfg["view"],
                    )
                    for t in range(count)
                ]
                e["points"] = np.stack([p for p, m in pairs])
                e["mask"] = np.stack([m for p, m in pairs])
            eid = len(self.episodes)
            self.episodes.append(e)
            self.index.extend((eid, t) for t in range(count))
            self.identities.append(
                dict(path=record["path"], sha256=record["sha256"], frames=count)
            )

    def batch(self, rng, size, norm, aug_rng=None):
        samples = []
        indices = rng.integers(len(self.index), size=size)
        for ix in indices:
            eid, t = self.index[ix]
            e = self.episodes[eid]
            future = np.minimum(
                t + np.arange(self.cfg["horizon"]), len(e["action"]) - 1
            )
            p, m = e["points"][t], e["mask"][t]
            if aug_rng is not None:
                condition = augmentation_condition(
                    aug_rng, self.cfg["augmentation_recipe"]
                )
                if condition["family"] != "nominal":
                    packet = {
                        c: {k: v[t] for k, v in e["cameras"][c].items()}
                        for c in CAMERAS
                    }
                    p, m = fuse(perturb(packet, condition, aug_rng), self.cfg["view"])
            samples.append(
                dict(
                    state=history_state(e["state"], t, self.cfg["history"], norm),
                    instruction=e["instruction_tokens"][t],
                    action=(e["action"][future] - norm["action_mean"])
                    / norm["action_std"],
                    physical_action=e["action"][future],
                    action_mask=t + np.arange(self.cfg["horizon"]) < len(e["action"]),
                    points=p,
                    point_mask=m,
                )
            )
        b = {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}
        b.update(
            action_mean=torch.from_numpy(norm["action_mean"]),
            action_std=torch.from_numpy(norm["action_std"]),
        )
        return b


def live_batch(packet, history, cfg, norm):
    p, m = fuse(packet, cfg.get("view", "fusion"))
    states = np.stack(history[-cfg["history"] :])
    b = dict(
        state=history_state(states, len(states) - 1, cfg["history"], norm),
        instruction=np.array([1, 0, 1, 0], np.float32),
        points=p,
        point_mask=m,
    )
    return {k: torch.from_numpy(np.asarray(v)[None]) for k, v in b.items()}
