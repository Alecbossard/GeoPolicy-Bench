"""Shared authentic LeRobot adapter; run inside isolated .venv-vla only."""

import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch

os.environ.setdefault("HF_HOME", str(Path("artifacts/hf_cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def load_policy():
    from huggingface_hub import snapshot_download
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    revisions = json.loads(Path("configs/pretrained_revisions.json").read_text())
    base_revision = revisions["lerobot/smolvla_base"]["sha"]
    config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base", revision=base_revision)
    config.device = "cuda"
    config.chunk_size = 8
    config.n_action_steps = 1
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    config.load_vlm_weights = False
    config.vlm_model_name = snapshot_download(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision=revisions["HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]["sha"],
        allow_patterns=["*.json", "*.txt", "*.model", "tokenizer*"],
    )
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (23,)),
        "observation.images.fixed": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    policy = SmolVLAPolicy.from_pretrained(
        "lerobot/smolvla_base", config=config, revision=base_revision, strict=True
    ).cuda()
    return policy, config


def restore_adapter(policy, saved, checkpoint_path):
    """Adapter deltas require the exact pinned frozen base and complete key set."""
    revisions = saved.get("pretrained_revisions")
    if revisions is None:
        sidecar = Path(checkpoint_path).parent / "base_revisions.json"
        if not sidecar.exists():
            raise ValueError("Legacy adapter requires its base_revisions.json sidecar")
        revisions = json.loads(sidecar.read_text())
    current = json.loads(Path("configs/pretrained_revisions.json").read_text())
    if {k: v["sha"] for k, v in revisions.items()} != {k: v["sha"] for k, v in current.items()}:
        raise ValueError(
            "Frozen base/tokenizer revision mismatch; do not combine unrelated adapter/base weights"
        )
    if saved["base_model"] != "lerobot/smolvla_base":
        raise ValueError("Unexpected adapter base")
    expected = {name for name, p in policy.named_parameters() if p.requires_grad}
    if set(saved["trainable"]) != expected:
        raise ValueError("Adapter must include every trainable action-expert/projection parameter")
    policy.load_state_dict(saved["trainable"], strict=False)


class VLADataset:
    def __init__(self, root, split="train", limit=200):
        self.episodes = []
        self.index = []
        for path in sorted(Path(root).glob("episode_*.h5")):
            with h5py.File(path) as f:
                m = json.loads(f.attrs["metadata"])
                if m["split"] != split or not m["success"]:
                    continue
                state = f["state"][:]
                action = f["action"][:]
                i = len(self.episodes)
                self.episodes.append(
                    {"path": path, "state": state, "action": action, "metadata": m}
                )
                self.index.extend((i, t) for t in range(len(state)))
                if limit and len(self.episodes) >= limit:
                    break
        if not self.episodes:
            raise ValueError("No VLA episodes")

    def stats(self):
        out = {}
        for key, name in [("state", "observation.state"), ("action", "action")]:
            x = np.concatenate([e[key] for e in self.episodes])
            out[name] = {
                "mean": torch.tensor(x.mean(0)),
                "std": torch.tensor(np.maximum(x.std(0), 0.01)),
            }
        return out

    def sample(self, rng):
        eid, t = self.index[int(rng.integers(len(self.index)))]
        e = self.episodes[eid]
        future = np.minimum(t + np.arange(8), len(e["action"]) - 1)
        with h5py.File(e["path"]) as f:
            raw = {
                "observation.state": torch.tensor(e["state"][t]),
                "action": torch.tensor(e["action"][future])[None],
                "actions_id_pad": torch.tensor(t + np.arange(8) >= len(e["action"]))[None],
                "observation.images.fixed": torch.tensor(
                    f["agentview/rgb"][t].transpose(2, 0, 1).astype(np.float32) / 255
                ),
                "observation.images.wrist": torch.tensor(
                    f["robot0_eye_in_hand/rgb"][t].transpose(2, 0, 1).astype(np.float32) / 255
                ),
                "task": e["metadata"]["instruction"],
            }
        return raw


def processed_batch(dataset, rng, pre):
    raw = dataset.sample(rng)
    batch = pre(raw)
    # Upstream forward uses this key; preserve mask even if processor converters
    # omit an unrecognized auxiliary field.
    batch["actions_id_pad"] = raw["actions_id_pad"].to("cuda")
    return batch
