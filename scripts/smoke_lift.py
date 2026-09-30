import json
import argparse
import time
from pathlib import Path
import numpy as np
from geopolicy.runtime import setup

setup()
import mujoco
import robosuite as suite
import torch
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_real_depth_map
import imageio.v3 as iio

p = argparse.ArgumentParser()
p.add_argument("--out", default="artifacts/smoke_lift")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    camera_depths=True,
    controller_configs=load_composite_controller_config(controller="BASIC"),
    control_freq=20,
    horizon=120,
    reward_shaping=True,
)
obs = env.reset()
print("action_dim", env.action_dim, "eef", obs["robot0_eef_pos"], flush=True)
depth = get_real_depth_map(env.sim, obs["agentview_depth"])
assert np.isfinite(depth).all() and (depth > 0).all()
iio.imwrite(out / "fixed.png", obs["agentview_image"][::-1])
iio.imwrite(out / "wrist.png", obs["robot0_eye_in_hand_image"][::-1])
np.save(out / "depth_m.npy", depth)
initial = obs["robot0_eef_pos"].copy()
for i in range(120):
    action = np.zeros(env.action_dim)
    action[0] = 0.15 if i < 10 else 0
    obs, reward, done, info = env.step(action)
    if i == 9:
        displacement = obs["robot0_eef_pos"] - initial
        print("displacement", displacement, flush=True)
assert displacement[0] > 0.01
assert done
# Gate for actual CUDA backward/optimizer and optimizer-state memory.
device = "cuda" if torch.cuda.is_available() else "cpu"
net = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.ReLU(), torch.nn.Linear(128, 7)).to(
    device
)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
loss = net(torch.randn(4, 32, device=device)).square().mean()
loss.backward()
opt.step()
report = {
    "mujoco": mujoco.__version__,
    "robosuite": suite.__version__,
    "torch": torch.__version__,
    "device": device,
    "cuda_runtime": torch.version.cuda,
    "wall_seconds": time.perf_counter() - start,
    "action_dim": env.action_dim,
    "eef_displacement": displacement.tolist(),
    "depth_min_max_m": [float(depth.min()), float(depth.max())],
    "peak_torch_vram": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
}
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
env.close()
