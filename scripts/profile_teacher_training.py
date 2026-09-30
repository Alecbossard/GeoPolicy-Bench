"""A non-selected1024-transition CPU continuation solely for resource profiling."""

import json
import os
import threading
import time
from pathlib import Path
import psutil
import torch
from stable_baselines3 import PPO
from geopolicy.teacher import TeacherEnv
from geopolicy.checkpoint import random_state

torch.set_num_threads(2)
out = Path("artifacts/teacher_resource_profile")
out.mkdir(parents=True, exist_ok=True)
env = TeacherEnv(0)
model = PPO.load("artifacts/teacher_selected.zip", env=env, device="cpu")
before = torch.cat([p.detach().flatten() for p in model.policy.parameters()]).clone()
process = psutil.Process(os.getpid())
samples = []
stop = threading.Event()


def sample():
    while not stop.wait(0.2):
        memory = process.memory_info()
        samples.append(
            {
                "time": time.time(),
                "rss_bytes": memory.rss,
                "private_bytes": getattr(memory, "private", None),
                "ram_available_bytes": psutil.virtual_memory().available,
            }
        )


thread = threading.Thread(target=sample, daemon=True)
thread.start()
start = time.perf_counter()
initial = model.num_timesteps
try:
    model.learn(total_timesteps=1024, reset_num_timesteps=False)
finally:
    stop.set()
    thread.join()
seconds = time.perf_counter() - start
steps = model.num_timesteps - initial
model.save(out / "latest")
torch.save(random_state(), out / "random.pt")
after = torch.cat([p.detach().flatten() for p in model.policy.parameters()])
report = {
    "scope": "CPU optimizer+training physics, auxiliary non-selected continuation; main teacher/data/checkpoints unchanged",
    "transitions": steps,
    "wall_seconds": seconds,
    "transitions_per_second": steps / seconds,
    "peak_process_rss_bytes": max(s["rss_bytes"] for s in samples),
    "peak_process_private_bytes": max(s["private_bytes"] or 0 for s in samples),
    "minimum_system_ram_available_bytes": min(s["ram_available_bytes"] for s in samples),
    "device": "cpu",
    "torch_threads": 2,
    "parameter_change_l2": float((after - before).norm()),
    "training_scene_seed_range": "[0,100000), no test scenes",
    "checkpoint_selected_for_benchmark": False,
}
(out / "resources.json").write_text(json.dumps(samples, indent=2))
(out / "report.json").write_text(json.dumps(report, indent=2))
Path("results/teacher_resource_profile.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
env.close()
