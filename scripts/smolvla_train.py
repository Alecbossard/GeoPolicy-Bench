import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from vla_common import load_policy, VLADataset, processed_batch, restore_adapter
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="artifacts/dataset")
p.add_argument("--out", default="artifacts/smolvla_s0")
p.add_argument("--updates", type=int, default=500)
p.add_argument("--accumulation", type=int, default=4)
p.add_argument("--resume")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
rng = np.random.default_rng(0)
train = VLADataset(args.dataset)
val = VLADataset(args.dataset, "validation", None)
stats = train.stats()
policy, config = load_policy()
pre, _ = make_smolvla_pre_post_processors(config, stats)
named = {k: v for k, v in policy.named_parameters() if v.requires_grad}
optimizer = torch.optim.AdamW(named.values(), lr=1e-4, weight_decay=1e-10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates, eta_min=1e-5)
first = 1
best = float("inf")
logs = (
    json.loads((out / "learning_curve.json").read_text())
    if (out / "learning_curve.json").exists()
    else []
)
if args.resume:
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    if ckpt["scheduler"]["T_max"] != args.updates or ckpt["accumulation"] != args.accumulation:
        raise ValueError("Resume must retain original update budget and accumulation")
    restore_adapter(policy, ckpt, args.resume)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    stats = ckpt["stats"]
    pre, _ = make_smolvla_pre_post_processors(config, stats)
    first = ckpt["update"] + 1
    rng.bit_generator.state = ckpt["batch_rng"]
    random.setstate(ckpt["python_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    torch.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    best = ckpt["best_validation"]
probe_name = next(iter(named))
initial_probe = named[probe_name].detach().cpu().clone()
start = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
for update in range(first, args.updates + 1):
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(args.accumulation):
        batch = processed_batch(train, rng, pre)
        loss, metrics = policy(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SmolVLA loss")
        (loss / args.accumulation).backward()
        total += float(loss) / args.accumulation
    torch.nn.utils.clip_grad_norm_(named.values(), 10)
    optimizer.step()
    scheduler.step()
    if update % 10 == 0 or update == first:
        logs.append({"update": update, "loss": total, "wall_seconds": time.perf_counter() - start})
        print(json.dumps(logs[-1]), flush=True)
    if update % 100 == 0 or update == args.updates:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        torch.manual_seed(1042)
        torch.cuda.manual_seed_all(1042)
        val_rng = np.random.default_rng(1042)
        policy.eval()
        values = []
        with torch.no_grad():
            for _ in range(20):
                values.append(float(policy(processed_batch(val, val_rng, pre))[0]))
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        score = float(np.mean(values))
        logs[-1]["validation_loss"] = score
        checkpoint = {
            "trainable": {k: v.detach().cpu() for k, v in named.items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stats": stats,
            "update": update,
            "best_validation": min(best, score),
            "batch_rng": rng.bit_generator.state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "base_model": "lerobot/smolvla_base",
            "config": str(config),
            "accumulation": args.accumulation,
            "pretrained_revisions": json.loads(
                Path("configs/pretrained_revisions.json").read_text()
            ),
            "training_config": {
                "updates": args.updates,
                "batch_size": 1,
                "accumulation": args.accumulation,
                "seed": 0,
                "demonstrations": len(train.episodes),
            },
        }
        temp = out / "latest.tmp"
        torch.save(checkpoint, temp)
        temp.replace(out / "latest.pt")
        if score < best:
            best = score
            best_temp = out / "best.tmp"
            torch.save(checkpoint, best_temp)
            best_temp.replace(out / "best.pt")
        (out / "learning_curve.json").write_text(json.dumps(logs, indent=2))
        print("SmolVLA validation", update, score, flush=True)
manifest = {
    "base_model": "lerobot/smolvla_base",
    "updates": args.updates,
    "batch_size": 1,
    "accumulation": args.accumulation,
    "demonstrations": len(train.episodes),
    "trainable_parameters": sum(p.numel() for p in named.values()),
    "wall_seconds": time.perf_counter() - start,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    "best_validation_loss": best,
    "probe_parameter_change_l2": float((named[probe_name].detach().cpu() - initial_probe).norm()),
    "modalities": "two RGB images resized/padded to512, robot state, instruction text; no depth",
    "recipe": "frozen pretrained VLM, action expert and state/action projections fine-tuned",
    "closed_loop_evaluated": False,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
