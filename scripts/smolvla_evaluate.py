"""One process for authentic VLA + same unmodified Panda OSC simulation."""

import argparse
import json
import csv
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
import numpy as np
import torch
from vla_common import load_policy, restore_adapter
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from geopolicy.environment import SelectPlace, CAMERAS
from geopolicy.sensors import student_state, camera_packet
from geopolicy.evaluation import rgb_packet, CONDITIONS
from geopolicy.io import save_json
from resource_watch import ResourceWatch

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="artifacts/smolvla_s0/best.pt")
p.add_argument("--out", default="artifacts/smolvla_validation")
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--first-seed", type=int, default=100100)
p.add_argument("--conditions", nargs="+", default=["nominal"])
p.add_argument("--execute-steps", type=int, default=2)
p.add_argument("--videos", type=int, default=2)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
watch = ResourceWatch(out)
torch.set_num_threads(2)
identity = {
    "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
    "execute_steps": a.execute_steps,
    "first_seed": a.first_seed,
    "pretrained_revisions": json.loads(Path("configs/pretrained_revisions.json").read_text()),
}
if a.first_seed >= 200000:
    assert json.loads(Path("configs/benchmark_protocol.json").read_text())["frozen"]
identity_path = out / "evaluation_identity.json"
if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
    raise ValueError("Existing VLA evaluation has a different checkpoint/base/recipe")
identity_path.write_text(json.dumps(identity, indent=2))
saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
# Full training checkpoints are preserved on disk; inference does not need
# their large AdamW moment tensors while constructing the frozen base.
for training_only in ["optimizer", "scheduler", "random", "batch_rng"]:
    saved.pop(training_only, None)
policy, config = load_policy()
restore_adapter(policy, saved, a.checkpoint)
saved.pop("trainable")  # Already copied into the authentic policy parameters.
policy.eval()
pre, post = make_smolvla_pre_post_processors(config, saved["stats"])
env = SelectPlace(cameras=True)
rows = json.loads((out / "rollouts.json").read_text()) if (out / "rollouts.json").exists() else []
completed = {(r["condition"], r["scene_seed"]) for r in rows}
for condition in a.conditions:
    condition_start = time.monotonic()
    if condition not in CONDITIONS:
        raise ValueError(condition)
    for seed in range(a.first_seed, a.first_seed + a.episodes):
        if (condition, seed) in completed:
            continue
        obs = env.reset_scene(seed)
        policy.reset()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        pre_times = []
        policy_times = []
        queue = []
        frames = []
        start = time.perf_counter()
        for step in range(env.horizon):
            watch.sample()
            if time.monotonic() - condition_start > 3600:
                env.close()
                raise TimeoutError("VLA condition exceeded3600s; completed episodes preserved")
            if not queue:
                t = time.perf_counter()
                packet = rgb_packet(obs, condition, rng)
                raw = {
                    "observation.state": torch.tensor(student_state(obs)),
                    "task": env.instruction,
                    "observation.images.fixed": torch.tensor(
                        packet[CAMERAS[0]]["rgb"].transpose(2, 0, 1).astype(np.float32) / 255
                    ),
                    "observation.images.wrist": torch.tensor(
                        packet[CAMERAS[1]]["rgb"].transpose(2, 0, 1).astype(np.float32) / 255
                    ),
                }
                batch = pre(raw)
                torch.cuda.synchronize()
                pre_times.append(time.perf_counter() - t)
                t = time.perf_counter()
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(batch)[0, : a.execute_steps]
                std = saved["stats"]["action"]["std"].to(chunk.device)
                mean = saved["stats"]["action"]["mean"].to(chunk.device)
                actions = (chunk[:, :7] * std + mean).clamp(-1, 1).cpu().numpy()
                torch.cuda.synchronize()
                policy_times.append(time.perf_counter() - t)
                queue = list(actions)
            if seed - a.first_seed < a.videos:
                frames.append(np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], axis=1))
            obs, _, done, info = env.step(queue.pop(0))
            if done or info["success"]:
                break
        row = {
            "mode": "smolvla",
            "training_seed": 0,
            "scene_seed": seed,
            "condition": condition,
            "checkpoint": a.checkpoint,
            "instruction": env.instruction,
            "inference_device": "cuda",
            "execute_steps": a.execute_steps,
            "steps": step + 1,
            "wall_seconds": time.perf_counter() - start,
            "simulation_seconds": (step + 1) / 20,
            "preprocess_p50_ms": float(np.percentile(pre_times, 50) * 1000),
            "preprocess_p95_ms": float(np.percentile(pre_times, 95) * 1000),
            "policy_p50_ms": float(np.percentile(policy_times, 50) * 1000),
            "policy_p95_ms": float(np.percentile(policy_times, 95) * 1000),
            "total_p50_ms": float(np.percentile(np.array(pre_times) + policy_times, 50) * 1000),
            "total_p95_ms": float(np.percentile(np.array(pre_times) + policy_times, 95) * 1000),
            **info,
        }
        rows.append(row)
        save_json(out / "rollouts.json", rows)
        with (out / "rollouts.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if frames:
            import imageio.v2 as iio

            iio.mimwrite(
                out
                / f"smolvla_{condition}_{seed}_{'success' if info['success'] else 'failure'}.mp4",
                frames,
                fps=20,
            )
env.close()
