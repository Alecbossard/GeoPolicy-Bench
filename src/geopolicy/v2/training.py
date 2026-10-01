"""Configured serial training with matched sampling, validation and exact resume."""

import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from geopolicy.checkpoint import save, load, random_state, restore_random
from geopolicy.io import save_json
from .data import Dataset
from .models import build_model
from .resources import ResourceWatch
from .config import file_hash, json_hash


def train(recipe, cfg, out, resume=None):
    assert cfg["recipe_sha256"] == json_hash(recipe), "Training recipe mismatch"
    assert cfg["dataset_manifest_sha256"] == file_hash(recipe["dataset_manifest"])
    assert cfg["normalization_sha256"] == file_hash(cfg["normalization_path"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    watch = ResourceWatch(out, recipe["resource_limits"])
    watch.sample(force=True)
    data = Dataset(recipe, cfg)
    val = Dataset(recipe, cfg, "validation")
    norm = data.normalization()
    legacy = torch.load(
        "artifacts/main_runs/fusion_s0/best.pt", map_location="cpu", weights_only=False
    )["normalization"]
    for key in norm:
        np.testing.assert_array_equal(norm[key], legacy[key])
    device = "cuda"
    net = build_model(cfg, norm).to(device)
    ema = copy.deepcopy(net)
    train_cfg = recipe["training"]
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg["updates"], eta_min=train_cfg["minimum_learning_rate"]
    )
    start_update = 0
    best = float("inf")
    curve = (
        json.loads((out / "learning_curve.json").read_text())
        if (out / "learning_curve.json").exists()
        else []
    )
    if resume:
        saved = load(resume, net, optimizer, scheduler)
        assert saved["config"] == cfg, "Resume recipe mismatch"
        norm = saved["normalization"]
        ema.load_state_dict(saved["extra"]["ema"])
        rng.bit_generator.state = saved["extra"]["batch_rng"]
        start_update = saved["progress"]["update"]
        best = saved["progress"]["best_validation"]
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for update in range(start_update + 1, cfg["updates"] + 1):
        watch.sample()
        if time.monotonic() - start > recipe["resource_limits"]["training_timeout_seconds"]:
            raise TimeoutError("Training job timeout")
        batch = {k: v.cuda() for k, v in data.batch(rng, cfg["batch_size"], norm).items()}
        net.train()
        loss, metrics = net.loss(batch)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for e, n in zip(ema.parameters(), net.parameters()):
                e.lerp_(n, 1 - train_cfg["ema_decay"])
        if update % 100 == 0 or update == 1:
            curve.append(
                {
                    "update": update,
                    "training_loss": float(loss),
                    "seconds": time.monotonic() - start,
                    **metrics,
                }
            )
            print(json.dumps(curve[-1]), flush=True)
        if update % train_cfg["validation_interval"] == 0 or update == cfg["updates"]:
            state = random_state()
            torch.manual_seed(10042)
            torch.cuda.manual_seed_all(10042)
            vrng = np.random.default_rng(10042)
            ema.eval()
            with torch.no_grad():
                scores = [
                    float(
                        ema.loss(
                            {
                                k: v.cuda()
                                for k, v in val.batch(vrng, cfg["batch_size"], norm).items()
                            }
                        )[0]
                    )
                    for _ in range(train_cfg["validation_batches"])
                ]
            restore_random(state)
            score = float(np.mean(scores))
            curve[-1]["validation_loss"] = score
            progress = {"update": update, "best_validation": min(best, score)}
            extra = {"ema": ema.state_dict(), "batch_rng": rng.bit_generator.state}
            save(out / "latest.pt", net, optimizer, scheduler, norm, cfg, progress, extra)
            if update == cfg["updates"] - 500:
                save(out / "resume_probe.pt", net, optimizer, scheduler, norm, cfg, progress, extra)
            if score < best:
                best = score
                save(out / "best.pt", net, optimizer, scheduler, norm, cfg, progress, extra)
            save_json(out / "learning_curve.json", curve)
    watch.sample(force=True)
    manifest = {
        "config": cfg,
        "parameters": sum(p.numel() for p in net.parameters()),
        "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
        "optimization_seconds": time.monotonic() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "best_validation_loss": best,
        "training_episode_ids": [e["episode_id"] for e in data.episodes],
        "validation_episode_ids": [e["episode_id"] for e in val.episodes],
    }
    save_json(out / "manifest.json", manifest)
    return manifest
