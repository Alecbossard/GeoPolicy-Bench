"""Shared recorded/live inputs: only RGB-D, copied robot state and fixed labels."""
import json
from pathlib import Path
import numpy as np
import torch
from geopolicy.data import Episodes
from geopolicy.v2.data import history_state
from geopolicy.sensors import camera_packet, fuse_points, student_state
from geopolicy.environment import CAMERAS
from .common import ROOT, sha


class Data(Episodes):
    def __init__(self, cfg, split):
        super().__init__(ROOT / "artifacts/v3/single_dataset", split, "fusion",
                         cfg["horizon"], limit=cfg.get("limit"))
        manifest_sha = cfg.get("dataset_manifest_sha256")
        manifest = (ROOT / "configs/v3/datasets" / f"{manifest_sha}.json") if manifest_sha else (ROOT / "configs/v3/single_dataset.json")
        records = json.loads(manifest.read_text())
        selected = [r for r in records if r["split"] == split and r["success"]]
        if cfg.get("limit"):
            selected = selected[:cfg["limit"]]
        # A later collection may add episodes. Match only the immutable run manifest.
        assert len(selected) <= len(self.episodes)
        self.episodes = self.episodes[:len(selected)]
        self.identities = []
        self.index = []
        for eid, (episode, record) in enumerate(zip(self.episodes, selected)):
            assert sha(ROOT / record["path"]) == record["sha256"]
            frames = record["frames"] if cfg["continued"] else record["original_frames"]
            assert frames is not None
            for key in episode:
                episode[key] = episode[key][:frames]
            self.index.extend((eid, t) for t in range(frames))
            self.identities.append(dict(path=record["path"], sha256=record["sha256"], frames=frames))
        self.cfg = cfg

    def batch(self, rng, batch_size, norm):
        samples = []
        for ix in rng.integers(len(self.index), size=batch_size):
            eid, t = self.index[ix]
            episode = self.episodes[eid]
            future = np.minimum(t + np.arange(self.horizon), len(episode["action"]) - 1)
            samples.append(dict(state=history_state(episode["state"], t, self.cfg["history"], norm),
                                instruction=episode["instruction_tokens"][t],
                                action=(episode["action"][future] - norm["action_mean"]) / norm["action_std"],
                                physical_action=episode["action"][future],
                                action_mask=t + np.arange(self.horizon) < len(episode["action"]),
                                points=episode["points"][t], point_mask=episode["mask"][t]))
        batch = {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}
        batch.update(action_mean=torch.from_numpy(norm["action_mean"]), action_std=torch.from_numpy(norm["action_std"]))
        return batch


def live(env, obs, history, cfg, norm):
    packet = camera_packet(env, obs)
    points, mask = fuse_points([(packet[c]["points"], packet[c]["mask"]) for c in CAMERAS])
    states = np.stack(history[-cfg["history"]:])
    batch = dict(state=history_state(states, len(states) - 1, cfg["history"], norm),
                 instruction=np.array([1, 0, 1, 0], np.float32), points=points, point_mask=mask)
    batch = {k: torch.from_numpy(np.asarray(v)[None]) for k, v in batch.items()}
    batch.update(action_mean=torch.from_numpy(norm["action_mean"]), action_std=torch.from_numpy(norm["action_std"]))
    return batch
