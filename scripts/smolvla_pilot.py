"""Run only with .venv-vla; authentic pretrained model + trainable action expert."""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("artifacts/hf_cache").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import h5py
import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

out = Path("artifacts/smolvla_pilot")
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
report = {
    "status": "started",
    "pretrained": "lerobot/smolvla_base",
    "lerobot_version": "0.4.4",
    "batch_size": 1,
}
(out / "report.json").write_text(json.dumps(report, indent=2))
start = time.perf_counter()
try:
    config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
    config.device = "cuda"
    config.chunk_size = 8
    config.n_action_steps = 1
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    # The full SmolVLA checkpoint supplies VLM weights; avoid a redundant backbone
    # weight load. Require strict checkpoint loading so no random backbone remains.
    config.load_vlm_weights = False
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (23,)),
        "observation.images.fixed": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    print("Loading authentic pretrained SmolVLA", flush=True)
    policy = SmolVLAPolicy.from_pretrained(
        "lerobot/smolvla_base", config=config, strict=True
    ).cuda()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    with h5py.File("artifacts/data_smoke/episode_000000.h5") as f:
        state = f["state"][:]
        action = f["action"][:]
        stats = {
            "observation.state": {
                "mean": torch.tensor(state.mean(0)),
                "std": torch.tensor(np.maximum(state.std(0), 0.01)),
            },
            "action": {
                "mean": torch.tensor(action.mean(0)),
                "std": torch.tensor(np.maximum(action.std(0), 0.01)),
            },
        }
        frame = 40
        raw = {
            "observation.state": torch.tensor(state[frame]),
            "action": torch.tensor(action[frame : frame + 8])[None],
            "observation.images.fixed": torch.tensor(
                f["agentview/rgb"][frame].transpose(2, 0, 1).astype(np.float32) / 255
            ),
            "observation.images.wrist": torch.tensor(
                f["robot0_eye_in_hand/rgb"][frame].transpose(2, 0, 1).astype(np.float32) / 255
            ),
            "task": json.loads(f.attrs["metadata"])["instruction"],
        }
    pre, _ = make_smolvla_pre_post_processors(config, stats)
    batch = pre(raw)
    print(
        "Preprocessed shapes",
        {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in batch.items()},
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    before = [p.detach().clone() for p in parameters[:2]]
    policy.train()
    loss, metrics = policy(batch)
    print("Forward loss", float(loss), flush=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 10)
    optimizer.step()
    torch.cuda.synchronize()
    delta = sum(float((p.detach() - b).norm()) for p, b in zip(parameters[:2], before))
    free, total = torch.cuda.mem_get_info()
    report.update(
        status="passed",
        loss=float(loss),
        metrics=metrics,
        parameter_change_l2_probe=delta,
        total_parameters=sum(p.numel() for p in policy.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        free_bytes=free,
        total_vram_bytes=total,
        elapsed_seconds=time.perf_counter() - start,
        resize_imgs_with_padding=config.resize_imgs_with_padding,
        margin_ok=free > 1024**3,
        train_expert_only=config.train_expert_only,
    )
    assert delta > 0
    torch.save(
        {
            "trainable": {
                k: v.cpu()
                for k, v in policy.state_dict().items()
                if k in dict(policy.named_parameters())
                and dict(policy.named_parameters())[k].requires_grad
            },
            "optimizer": optimizer.state_dict(),
            "stats": stats,
            "config": config.to_dict() if hasattr(config, "to_dict") else str(config),
        },
        out / "pilot_step.pt",
    )
except Exception as e:
    report.update(
        status="failed",
        error_type=type(e).__name__,
        error=str(e),
        elapsed_seconds=time.perf_counter() - start,
    )
    if torch.cuda.is_available():
        report.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    raise
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
