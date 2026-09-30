"""Validation-only controls; test remains locked until protocol freeze."""

import json
import argparse
import csv
from pathlib import Path
import torch
from stable_baselines3 import PPO
from geopolicy.teacher import TeacherEnv, evaluate

torch.set_num_threads(2)
p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--first-seed", type=int, default=100100)
p.add_argument("--out", default="artifacts/teacher_controls.json")
args = p.parse_args()
if args.first_seed >= 200000:
    assert json.loads(Path("configs/benchmark_protocol.json").read_text())["frozen"]
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
env = TeacherEnv(0)
initial = PPO(
    "MlpPolicy",
    env,
    seed=0,
    device="cpu",
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-2.7),
)
bc = PPO.load("artifacts/teacher_pilot/bc_initial.zip", device="cpu")
ppo = PPO.load("artifacts/teacher_selected.zip", device="cpu")
out = {}
for name, model, kwargs in [
    ("initial", initial, {}),
    ("scripted_reference", None, {"reference": True}),
    ("bc_initial", bc, {}),
    ("ppo_selected", ppo, {}),
]:
    rows = evaluate(
        model, env.env, range(args.first_seed, args.first_seed + args.episodes), **kwargs
    )
    out[name] = rows
    print(name, sum(r["success"] for r in rows), "/", len(rows), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    flattened = [dict(control=k, **r) for k, v in out.items() for r in v]
    with Path(args.out).with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
env.close()
