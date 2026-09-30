"""BC-initialized PPO teacher; script/BC/PPO provenance remains separate."""

import json
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .environment import SelectPlace
from .checkpoint import random_state, restore_random


class TeacherEnv(gym.Env):
    def __init__(self, seed=0):
        super().__init__()
        self.env = SelectPlace(seed=seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self.counter = seed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Train scenes exclusively in [0, 100000); validation/test disjoint.
        self.counter = int(self.np_random.integers(0, 100000))
        self.env.reset_scene(self.counter)
        return self.env.teacher_state(), {}

    def step(self, action):
        _, r, done, info = self.env.step(action)
        return (
            self.env.teacher_state(),
            r,
            info["success"],
            bool(done and not info["success"]),
            info,
        )

    def close(self):
        self.env.close()


def evaluate(model, env, seeds, reference=False, random=False):
    rows = []
    rng = np.random.default_rng(42)
    for seed in seeds:
        env.reset_scene(seed)
        start = time.perf_counter()
        total = 0.0
        for step in range(env.horizon):
            if reference:
                action = env.reference_action()
            elif random:
                action = rng.uniform(-1, 1, 7)
            else:
                action, _ = model.predict(env.teacher_state(), deterministic=True)
            _, r, done, info = env.step(action)
            total += r
            if done or info["success"]:
                break
        rows.append(
            dict(
                scene_seed=seed,
                steps=step + 1,
                seconds=time.perf_counter() - start,
                episode_return=total,
                final_phase=env.phase,
                **info,
            )
        )
    return rows


class LogCheckpoint(BaseCallback):
    def __init__(self, out, interval=4096):
        super().__init__()
        self.out = Path(out)
        self.interval = interval
        self.start = time.perf_counter()
        self.episodes = []
        self.validation_env = SelectPlace(seed=0)
        self.best = -1.0
        self.validations = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes.append(
                    {
                        "transitions": self.num_timesteps,
                        "success": info.get("success", False),
                        "wall_seconds": time.perf_counter() - self.start,
                    }
                )
        return True

    def _on_rollout_start(self):
        # SB3 owns optimizer/learning-rate schedule; checkpoints save it in policy.
        if self.num_timesteps and self.num_timesteps % self.interval == 0:
            self.model.save(self.out / "latest")
            torch.save(random_state(), self.out / "random.pt")
            (self.out / "episodes.json").write_text(json.dumps(self.episodes, indent=2))
            rate = self.num_timesteps / (time.perf_counter() - self.start)
            print(
                f"PPO {self.num_timesteps} transitions, {rate:.1f}/s, recent success {np.mean([e['success'] for e in self.episodes[-20:]]) if self.episodes else 0:.2f}",
                flush=True,
            )
            rows = evaluate(self.model, self.validation_env, range(100000, 100020))
            score = float(np.mean([r["success"] for r in rows]))
            self.validations.append(
                {
                    "transitions": self.num_timesteps,
                    "success": score,
                    "rollouts": rows,
                    "optimization": {
                        k: float(v)
                        for k, v in self.model.logger.name_to_value.items()
                        if isinstance(v, (int, float, np.floating))
                    },
                }
            )
            (self.out / "validation_curve.json").write_text(json.dumps(self.validations, indent=2))
            if score > self.best:
                self.best = score
                self.model.save(self.out / "best")
            print(
                f"PPO validation at {self.num_timesteps}: {score:.2f}; best {self.best:.2f}",
                flush=True,
            )

    def _on_training_end(self):
        self.validation_env.close()


def train(out, bootstrap, steps=65536, seed=0, resume=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    env = TeacherEnv(seed)
    config = dict(
        seed=seed,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        learning_rate=1e-4,
        gamma=0.98,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-2.7),
        device="cpu",
    )
    if resume:
        model = PPO.load(resume, env=env, device="cpu")
        random_path = Path(resume).parent / "random.pt"
        if random_path.exists():
            restore_random(torch.load(random_path, weights_only=False))
    else:
        model = PPO("MlpPolicy", env, **config)
        # Explicit supervised initialization, never mislabeled as PPO learning.
        data = np.load(bootstrap)
        x, y = torch.tensor(data["state"]), torch.tensor(data["action"])
        bc = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
        for update in range(2500):
            idx = torch.randint(len(x), (256,))
            distribution = model.policy.get_distribution(x[idx])
            mean = distribution.distribution.mean
            loss = (mean - y[idx]).square().mean()
            bc.zero_grad()
            loss.backward()
            bc.step()
            if update % 500 == 0:
                print(f"BC update {update}, loss {loss.item():.5f}", flush=True)
        model.save(out / "bc_initial")
        initial = evaluate(model, env.env, range(100000, 100020))
        (out / "bc_validation.json").write_text(json.dumps(initial, indent=2))
        print("BC validation", np.mean([r["success"] for r in initial]), flush=True)
    before = torch.cat([v.detach().flatten() for v in model.policy.parameters()]).clone()
    callback = LogCheckpoint(out)
    model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=not bool(resume))
    model.save(out / "latest")
    torch.save(random_state(), out / "random.pt")
    after = torch.cat([v.detach().flatten() for v in model.policy.parameters()])
    validation = evaluate(model, env.env, range(100000, 100020))
    (out / "ppo_validation.json").write_text(json.dumps(validation, indent=2))
    manifest = {
        "config": config,
        "steps": model.num_timesteps,
        "parameter_change_l2": float((after - before).norm()),
        "bootstrap": str(bootstrap),
        "provenance": "scripted BC initialization followed by real on-policy PPO",
        "validation_success": float(np.mean([r["success"] for r in validation])),
        "resume_semantics": "optimizer and RNG restored; new episode at resume boundary (not a bit-exact interrupted physics rollout)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env.close()
    return manifest
