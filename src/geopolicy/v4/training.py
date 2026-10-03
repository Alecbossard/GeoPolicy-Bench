"""Bounded V3-matched BC training with independent, resumable augmentation RNG."""

import copy
import hashlib
import json
import random
import time
import numpy as np
import torch
from geopolicy.checkpoint import save, load
from geopolicy.v2.resources import ResourceWatch
from geopolicy.v3.models import build
from .common import ROOT, plan, sha, write, read
from .data import Data


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(view, augmented, seed, resume=False, stop_at=None, scope="pilot"):
    p = plan()
    recipe = p["training"]
    updates = recipe["updates"]
    name = f'{view}_{"aug" if augmented else "clean"}_s{seed}'
    directory = "main_runs" if scope == "main" else "runs"
    out = ROOT / "artifacts/v4" / directory / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        recipe,
        version=4,
        seed=seed,
        view=view,
        augmented=augmented,
        continued=True,
        task="single",
        device="cuda",
        dataset_manifest_sha256=sha(ROOT / "configs/v3/single_dataset.json"),
        augmentation_recipe=p["augmentation"],
    )
    ip = out / "config.json"
    if ip.exists():
        assert resume and read(ip) == cfg, "Run identity changed or --resume missing"
    else:
        assert not resume
    write(ip, cfg)
    write(ROOT / "configs/v4" / directory / f"{name}.json", cfg)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    aug_rng = np.random.default_rng(771000 + seed)
    data = Data(cfg, "train")
    val = Data(cfg, "validation")
    reference = torch.load(
        ROOT / "artifacts/v3/runs/bc_continued_history480_s0/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    norm = reference["normalization"]
    del reference
    net = build(cfg).cuda()
    ema = copy.deepcopy(net)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, updates, eta_min=recipe["minimum_learning_rate"]
    )
    initial = 0
    best = float("inf")
    curve = []
    if resume:
        saved = load(out / "latest.pt", net, optimizer, scheduler)
        assert (
            saved["config"] == cfg
            and saved["extra"]["data_identity"] == data.identities
        )
        norm = saved["normalization"]
        ema.load_state_dict(saved["extra"]["ema"])
        rng.bit_generator.state = saved["extra"]["batch_rng"]
        aug_rng.bit_generator.state = saved["extra"]["augmentation_rng"]
        initial = saved["progress"]["update"]
        best = saved["progress"]["best_loss"]
        curve = read(out / "curve.json")
    watch = ResourceWatch(out, p["resource_limits"])
    watch.sample(force=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    end = min(updates, stop_at or updates)
    assert end >= initial
    for update in range(initial + 1, end + 1):
        batch = {
            k: v.cuda()
            for k, v in data.batch(
                rng, recipe["batch_size"], norm, aug_rng if augmented else None
            ).items()
        }
        net.train()
        loss, metrics = net.loss(batch)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), net.parameters()):
                average.lerp_(current, 1 - recipe["ema_decay"])
        if update == 1 or update % 100 == 0:
            curve.append(
                dict(
                    update=update,
                    loss=float(loss.detach()),
                    seconds=time.perf_counter() - started,
                    **metrics,
                )
            )
            print(json.dumps(curve[-1]), flush=True)
        score = None
        if update % recipe["validation_interval"] == 0 or update == updates:
            vrng = np.random.default_rng(12042)
            errors = []
            ema.eval()
            with torch.inference_mode():
                for _ in range(recipe["validation_batches"]):
                    vb = {
                        k: v.cuda()
                        for k, v in val.batch(vrng, recipe["batch_size"], norm).items()
                    }
                    mask = vb["action_mask"][:, :, None]
                    errors.append(
                        float(
                            ((ema.predict(vb) - vb["action"]).abs() * mask).sum()
                            / (7 * mask.sum())
                        )
                    )
            score = float(np.mean(errors))
            curve[-1]["deployed_clean_validation_l1"] = score
        if update % recipe["checkpoint_interval"] == 0 or update == end:
            extra = dict(
                ema=ema.state_dict(),
                batch_rng=rng.bit_generator.state,
                augmentation_rng=aug_rng.bit_generator.state,
                data_identity=data.identities,
            )
            progress = dict(
                update=update, best_loss=min(best, score) if score is not None else best
            )
            save(
                out / "latest.pt", net, optimizer, scheduler, norm, cfg, progress, extra
            )
            if update % recipe["validation_interval"] == 0 or update == end:
                save(
                    out / f"step_{update}.pt",
                    net,
                    optimizer,
                    scheduler,
                    norm,
                    cfg,
                    progress,
                    extra,
                )
            if score is not None and score < best:
                best = score
                save(
                    out / "best.pt",
                    net,
                    optimizer,
                    scheduler,
                    norm,
                    cfg,
                    progress,
                    extra,
                )
            write(out / "curve.json", curve)
            watch.sample(force=True)
            print(
                f"Saved {name} update{update}; validationL1={score}; behavioral success is separate",
                flush=True,
            )
        if time.perf_counter() - started > p["resource_limits"]["cell_timeout_seconds"]:
            raise RuntimeError("Training time limit; resume latest saved checkpoint")
    manifest = dict(
        config=cfg,
        training_data=data.identities,
        actual_train_episodes=len(data.episodes),
        training_frames=len(data.index),
        validation_episodes=len(val.episodes),
        completed_update=end,
        resumed_from_update=initial,
        optimizer_updates_this_session=end - initial,
        elapsed_this_session_s=time.perf_counter() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        best_checkpoint_sha256=(
            sha(out / "best.pt") if (out / "best.pt").exists() else None
        ),
        ema_hash=tensor_hash(ema.state_dict()),
    )
    write(out / "manifest.json", manifest)
    write(
        ROOT
        / "results/v4"
        / ("training" if scope == "main" else "pilot_training")
        / f"{name}.json",
        dict(manifest=manifest, curve=curve),
    )
    return manifest
