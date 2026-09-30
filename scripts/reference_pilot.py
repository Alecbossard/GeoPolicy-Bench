import argparse
import json
import time
from pathlib import Path
import numpy as np
from geopolicy.environment import SelectPlace
from geopolicy.sensors import camera_packet, student_state

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=4)
p.add_argument("--cameras", action="store_true")
args = p.parse_args()
out = Path("artifacts/reference_pilot")
out.mkdir(parents=True, exist_ok=True)
env = SelectPlace(cameras=args.cameras)
records = []
states, actions = [], []
for seed in range(args.episodes):
    obs = env.reset_scene(seed)
    start = time.perf_counter()
    for step in range(env.horizon):
        a = env.reference_action()
        states.append(env.teacher_state())
        actions.append(a)
        if args.cameras and step in [0, 30, 60, 90, 120, 150]:
            import imageio.v3 as iio

            packet = camera_packet(env, obs)
            for camera in ["agentview", "robot0_eye_in_hand"]:
                iio.imwrite(out / f"s{seed}_t{step}_{camera}.png", packet[camera]["rgb"])
            if step == 0:
                np.savez_compressed(
                    out / f"s{seed}_sensor.npz", state=student_state(obs), **packet["agentview"]
                )
        obs, r, done, info = env.step(a)
        if info["success"]:
            break
    record = dict(
        seed=seed,
        instruction=env.instruction,
        steps=step + 1,
        seconds=time.perf_counter() - start,
        phase=env.phase,
        eef=obs["robot0_eef_pos"].tolist(),
        objects=env.object_positions().tolist(),
        **info,
    )
    print(json.dumps(record), flush=True)
    records.append(record)
np.savez_compressed(out / "bootstrap.npz", state=np.array(states), action=np.array(actions))
(out / "results.json").write_text(json.dumps(records, indent=2))
env.close()
