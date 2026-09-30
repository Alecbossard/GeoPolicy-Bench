"""Actual optimizer step while dual RGB-D rendering remains alive; disposable model."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import psutil

sys.path.insert(0, str(Path("src").resolve()))
from geopolicy.runtime import setup

setup()
import torch
from geopolicy.environment import SelectPlace

p = argparse.ArgumentParser()
p.add_argument("--vla", action="store_true")
a = p.parse_args()
torch.set_num_threads(2)
env = SelectPlace(cameras=True)
obs = env.reset_scene(1000)
assert all(obs[c + "_image"].shape == (128, 128, 3) for c in ["agentview", "robot0_eye_in_hand"])
reports = []
if a.vla:
    from vla_common import load_policy, VLADataset, processed_batch, restore_adapter
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    saved = torch.load("artifacts/smolvla_s0/best.pt", map_location="cpu", weights_only=False)
    model, config = load_policy()
    restore_adapter(model, saved, "artifacts/smolvla_s0/best.pt")
    pre, _ = make_smolvla_pre_post_processors(config, saved["stats"])
    dataset = VLADataset("artifacts/dataset", limit=1)
    batch = processed_batch(dataset, np.random.default_rng(0), pre)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-10
    )
    optimizer.load_state_dict(saved["optimizer"])
    # Optimizer moments are now live on the model device; discard duplicate CPU
    # checkpoint storage without altering the selected checkpoint on disk.
    saved.pop("optimizer")
    saved.pop("trainable")
    jobs = [("smolvla", model, optimizer, batch)]
else:
    from geopolicy.data import Episodes
    from geopolicy.policies import make_policy

    jobs = []
    # Load/execute one model at a time below; renderer stays alive throughout.
    for mode in ["act", "mono", "fusion"]:
        jobs.append((mode, None, None, None))
for mode, model, optimizer, batch in jobs:
    if model is None:
        saved = torch.load(
            f"artifacts/main_runs/{mode}_s0/best.pt", map_location="cpu", weights_only=False
        )
        cfg = saved["config"]
        model = make_policy(
            mode, prediction_type=cfg.get("prediction_type"), color_prior=cfg.get("color_prior")
        ).cuda()
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        optimizer.load_state_dict(saved["optimizer"])
        dataset = Episodes("artifacts/dataset", mode=mode, limit=1)
        batch = {
            k: v.cuda()
            for k, v in dataset.batch(np.random.default_rng(0), 32, saved["normalization"]).items()
        }
    probe = next(p for p in model.parameters() if p.requires_grad)
    before = probe.detach().float().clone()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    loss, _ = model(batch) if mode == "smolvla" else model.loss(batch)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    # This fresh camera observation is rendered with optimizer moments still allocated.
    obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    memory = psutil.Process(os.getpid()).memory_info()
    assert (
        float(gpu.stdout.strip().split(",")[1]) >= 1024
    ), "Combined gate requires at least1GiB GPU headroom"
    reports.append(
        {
            "mode": mode,
            "batch_size": 1 if mode == "smolvla" else 32,
            "horizon": 8,
            "loss": float(loss),
            "probe_delta_l2": float((probe.detach().float() - before).norm()),
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_used_free_mib_temperature_c": gpu.stdout.strip(),
            "process_rss_bytes": memory.rss,
            "process_private_bytes": getattr(memory, "private", None),
            "optimizer_states_loaded_and_live": True,
            "dual_rgbd_renderer_live": True,
            "selected_benchmark_checkpoint_modified": False,
        }
    )
    if mode != "smolvla":
        del model, optimizer, batch, dataset, saved, probe, before
        torch.cuda.empty_cache()
env.close()
path = Path(
    "results/combined_vla_resources.json" if a.vla else "results/combined_compact_resources.json"
)
path.write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
