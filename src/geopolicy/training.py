import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import Episodes
from .policies import make_policy
from .checkpoint import save, load


def train_student(
    dataset,
    out,
    mode="fusion",
    seed=0,
    updates=3000,
    batch_size=32,
    resume=None,
    limit=None,
    device=None,
    view_dropout=0,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    data = Episodes(dataset, "train", mode, limit=limit, view_dropout=view_dropout)
    data.dropout_rng = np.random.default_rng(seed + 10000)
    val = Episodes(dataset, "validation", mode)
    norm = data.normalization()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = make_policy(mode).to(device)
    ema = copy.deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=3e-5)
    config = dict(
        mode=mode,
        seed=seed,
        updates=updates,
        batch_size=batch_size,
        horizon=8,
        device=device,
        demonstration_episodes=len(data.episodes),
        dataset=str(dataset),
        point_budget=512,
        pretrained=False,
        ema_decay=0.995,
        inference_steps=10,
        execute_steps=2,
    )
    config["view_dropout"] = view_dropout
    config["prediction_type"] = "sample" if mode != "act" else None
    config["point_encoder_color_prior"] = (
        "40x normalized RGB chromatic contrast, dark-intensity attenuation, learned attention added; no GT"
    )
    config["color_prior"] = "chroma40" if mode != "act" else False
    start_update = 0
    best = float("inf")
    logs = []
    if (out / "learning_curve.json").exists():
        logs = json.loads((out / "learning_curve.json").read_text())
    if resume:
        checkpoint = load(resume, net, opt, scheduler)
        for key in [
            "mode",
            "seed",
            "device",
            "prediction_type",
            "color_prior",
            "horizon",
            "batch_size",
            "updates",
            "demonstration_episodes",
        ]:
            if checkpoint["config"].get(key) != config.get(key):
                raise ValueError(
                    f"Resume recipe mismatch for {key}; resume the original planned budget/recipe"
                )
        if checkpoint["config"].get("view_dropout", 0) != view_dropout:
            raise ValueError("Resume view-dropout recipe mismatch")
        norm = checkpoint["normalization"]
        start_update = checkpoint["progress"]["update"]
        ema.load_state_dict(checkpoint["extra"]["ema"])
        rng.bit_generator.state = checkpoint["extra"]["batch_rng"]
        if view_dropout:
            data.dropout_rng.bit_generator.state = checkpoint["extra"]["dropout_rng"]
        best = checkpoint["progress"].get("best_validation", best)
    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for update in range(start_update + 1, updates + 1):
        batch = {k: v.to(device) for k, v in data.batch(rng, batch_size, norm).items()}
        net.train()
        loss, metrics = net.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        opt.step()
        scheduler.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), net.parameters()):
                average.lerp_(current, 0.005)
        if update % 100 == 0 or update == 1:
            log = dict(
                update=update,
                loss=float(loss),
                seconds=time.perf_counter() - start,
                lr=scheduler.get_last_lr()[0],
                **metrics,
            )
            logs.append(log)
            print(json.dumps(log), flush=True)
        if update % 500 == 0 or update == updates:
            # Fixed independent validation batching/noise seed; global training RNG restored afterwards.
            state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if device == "cuda" else []
            torch.manual_seed(10042)
            if device == "cuda":
                torch.cuda.manual_seed_all(10042)
            validation_rng = np.random.default_rng(10042)
            ema.eval()
            vals = []
            with torch.no_grad():
                for _ in range(20):
                    b = {
                        k: v.to(device)
                        for k, v in val.batch(validation_rng, batch_size, norm).items()
                    }
                    l, _ = ema.loss(b)
                    vals.append(float(l))
            torch.set_rng_state(state)
            if cuda_state:
                torch.cuda.set_rng_state_all(cuda_state)
            score = float(np.mean(vals))
            logs[-1]["validation_loss"] = score
            extra = {"ema": ema.state_dict(), "batch_rng": rng.bit_generator.state}
            if view_dropout:
                extra["dropout_rng"] = data.dropout_rng.bit_generator.state
            progress = {"update": update, "best_validation": min(best, score)}
            save(out / "latest.pt", net, opt, scheduler, norm, config, progress, extra)
            if score < best:
                best = score
                save(out / "best.pt", net, opt, scheduler, norm, config, progress, extra)
            (out / "learning_curve.json").write_text(json.dumps(logs, indent=2))
            print(f"Validation {mode} seed {seed} update {update}: {score:.5f}", flush=True)
    manifest = {
        "config": config,
        "parameters": sum(p.numel() for p in net.parameters()),
        "wall_seconds": time.perf_counter() - start,
        "peak_torch_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "best_validation_loss": best,
        "checkpoints": [str(out / "best.pt"), str(out / "latest.pt")],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
