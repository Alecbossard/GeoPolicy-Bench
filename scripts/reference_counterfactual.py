"""Four instructions in each identical scene, actual teacher/reference controls."""

import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from geopolicy.environment import SelectPlace

assert json.loads(Path("configs/benchmark_protocol.json").read_text())["frozen"]
torch.set_num_threads(2)
model = PPO.load("artifacts/teacher_selected.zip", device="cpu")
env = SelectPlace(cameras=False)
rows = []
out = Path("results/raw/reference_counterfactual.json")
for seed in range(200100, 200110):
    geometry = None
    for control in ["scripted_reference", "ppo_selected"]:
        for object_id in [0, 1]:
            for goal_id in [0, 1]:
                env.reset_scene(seed, object_id, goal_id)
                initial = np.r_[env.object_positions().ravel(), env.goal_positions.ravel()]
                if geometry is None:
                    geometry = initial.copy()
                else:
                    np.testing.assert_array_equal(initial, geometry)
                for step in range(env.horizon):
                    action = (
                        env.reference_action()
                        if control == "scripted_reference"
                        else model.predict(env.teacher_state(), deterministic=True)[0]
                    )
                    _, _, done, info = env.step(action)
                    if done or info["success"]:
                        break
                rows.append(
                    {
                        "scene_seed": seed,
                        "object_id": object_id,
                        "goal_id": goal_id,
                        "control": control,
                        "instruction": env.instruction,
                        "initial_geometry_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                        "steps": step + 1,
                        **info,
                    }
                )
                out.write_text(json.dumps(rows, indent=2))
    print(
        seed,
        {
            control: sum(
                r["success"] for r in rows if r["scene_seed"] == seed and r["control"] == control
            )
            for control in ["scripted_reference", "ppo_selected"]
        },
        flush=True,
    )
env.close()
