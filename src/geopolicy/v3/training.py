"""Bounded, resumable FP32 V3 optimization; loss is never a success criterion."""
import copy
import json
import random
import time
import numpy as np
import torch
from geopolicy.checkpoint import save, load
from geopolicy.v2.resources import ResourceWatch
from .common import ROOT, sha, write
from .data import Data
from .models import build


def train(name, model="direct_bc", seed=0, updates=2000, limit=4, continued=False,
          history=1, binary=False, overfit=False, resume=False, task="single", view="fusion", prior=True):
    assert name.replace("_", "").isalnum()
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    out = ROOT / "artifacts/v3/runs" / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(version=3, model=model, seed=seed, updates=updates, batch_size=plan["batch_size"],
               limit=limit, continued=continued, history=history, binary_gripper=binary,
               overfit_diagnostic=overfit, horizon=plan["horizon"],
               execute_steps=plan["execute_steps"], color_prior="chroma40",
               device="cuda", learning_rate=plan["learning_rate"],
               dataset_manifest_sha256=sha(ROOT / "configs/v3" / ("single_dataset.json" if task == "single" else f"{task}_dataset.json")))
    dataset_class = Data
    if task != "single":
        from .stages import StageData
        dataset_class = StageData
        cfg.update(task=task, view=view)
        cfg["color_prior"] = "chroma40" if prior else False
    ip = out / "config.json"
    if ip.exists():
        if resume:
            cfg["dataset_manifest_sha256"] = json.loads(ip.read_text())["dataset_manifest_sha256"]
        assert json.loads(ip.read_text()) == cfg, "Run identity changed; choose a new name"
        assert resume, "Existing run requires --resume"
    else:
        assert not resume
    write(ip, cfg)
    write(ROOT / "configs/v3/runs" / f"{name}.json", cfg)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    data = dataset_class(cfg, "train")
    val = data if overfit else dataset_class(cfg, "validation")
    if continued:
        # Hold normalization fixed when isolating the addition of suffix samples.
        original = dataset_class(dict(cfg, continued=False), "train")
        norm = original.normalization()
        del original
    else:
        norm = data.normalization()
    net = build(cfg).cuda()
    ema = copy.deepcopy(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=3e-5)
    watch = ResourceWatch(out, plan["resource_limits"])
    watch.sample(force=True)
    initial = 0
    best = float("inf")
    curve = []
    if resume:
        saved = load(out / "latest.pt", net, optimizer, scheduler)
        assert saved["config"] == cfg and saved["extra"]["data_identity"] == data.identities
        norm = saved["normalization"]
        ema.load_state_dict(saved["extra"]["ema"])
        rng.bit_generator.state = saved["extra"]["batch_rng"]
        initial = saved["progress"]["update"]
        best = saved["progress"]["best_loss"]
        curve = json.loads((out / "curve.json").read_text())
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for update in range(initial + 1, updates + 1):
        # Safe stop occurs immediately after a saved checkpoint (<=500 updates).
        batch = {k: v.cuda() for k, v in data.batch(rng, cfg["batch_size"], norm).items()}
        net.train()
        loss, metrics = net.loss(batch)
        assert torch.isfinite(loss), "Nonfinite training loss"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), net.parameters()):
                average.lerp_(current, .005)
        if update == 1 or update % 100 == 0:
            curve.append(dict(update=update, loss=float(loss.detach()),
                              seconds=time.perf_counter() - started, **metrics))
            print(json.dumps(curve[-1]), flush=True)
        if update % 500 == 0 or update == updates:
            # Use deployed predictions rather than target-conditioned training paths.
            state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all()
            torch.manual_seed(12042)
            torch.cuda.manual_seed_all(12042)
            errors, raw_errors, signs = [], [], []
            vrng = np.random.default_rng(12042)
            net.eval()
            ema.eval()
            with torch.inference_mode():
                for _ in range(20):
                    vb = {k: v.cuda() for k, v in val.batch(vrng, cfg["batch_size"], norm).items()}
                    predicted = ema.predict(vb)
                    raw = net.predict(vb)
                    mask = vb["action_mask"][:, :, None]
                    errors.append(float(((predicted - vb["action"]).abs() * mask).sum() / (7 * mask.sum())))
                    raw_errors.append(float(((raw - vb["action"]).abs() * mask).sum() / (7 * mask.sum())))
                    physical = predicted * vb["action_std"] + vb["action_mean"]
                    signs.append(float((((physical[:, :, 6] > 0) == (vb["physical_action"][:, :, 6] > 0)) * mask[:, :, 0]).sum() / mask.sum()))
            torch.set_rng_state(state)
            torch.cuda.set_rng_state_all(cuda_state)
            score = float(np.mean(errors))
            curve[-1].update(deployed_normalized_l1=score, raw_normalized_l1=float(np.mean(raw_errors)),
                             gripper_sign_accuracy=float(np.mean(signs)))
            progress = dict(update=update, best_loss=min(best, score))
            extra = dict(ema=ema.state_dict(), batch_rng=rng.bit_generator.state,
                         data_identity=data.identities)
            save(out / "latest.pt", net, optimizer, scheduler, norm, cfg, progress, extra)
            save(out / f"step_{update}.pt", net, optimizer, scheduler, norm, cfg, progress, extra)
            if score < best:
                best = score
                save(out / "best.pt", net, optimizer, scheduler, norm, cfg, progress, extra)
            write(out / "curve.json", curve)
            watch.sample(force=True)
            print(f"Saved {name} update {update}; deployed L1 {score:.6f}; not a behavioral score", flush=True)
    manifest = dict(config=cfg, training_data=data.identities,
                    parameters=sum(p.numel() for p in net.parameters()),
                    elapsed_seconds=time.perf_counter() - started,
                    peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                    checkpoint_sha256=sha(out / "best.pt"))
    write(out / "manifest.json", manifest)
    write(ROOT / "results/v3/training" / f"{name}.json", dict(manifest=manifest, curve=curve))
    return manifest
