import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from geopolicy.policies import make_policy
from geopolicy.sensors import fuse_points

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="artifacts/fusion_pilot_v2_s0/best.pt")
p.add_argument("--out", default="artifacts/policy_profile.json")
a = p.parse_args()
s = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
cfg = s["config"]
norm = s["normalization"]
with h5py.File("artifacts/dataset/episode_001000.h5") as f:
    state = f["state"][30]
    tokens = f["instruction_tokens"][30]
    sensors = {}
    if cfg["mode"] == "act":
        rgb = (
            np.stack(
                [
                    f[c + "/rgb"][30, ::2, ::2].transpose(2, 0, 1)
                    for c in ["agentview", "robot0_eye_in_hand"]
                ]
            ).astype(np.float32)
            / 255
        )
        sensors["rgb"] = torch.tensor(rgb)[None]
    else:
        cameras = ["agentview", "robot0_eye_in_hand"] if cfg["mode"] == "fusion" else ["agentview"]
        views = [(f[c + "/points"][30], f[c + "/mask"][30]) for c in cameras]
        points, mask = fuse_points(views)
        sensors = {"points": torch.tensor(points)[None], "point_mask": torch.tensor(mask)[None]}
b = {
    "state": torch.tensor((state - norm["state_mean"]) / norm["state_std"])[None],
    "instruction": torch.tensor(tokens)[None],
    **sensors,
}
net = make_policy(
    cfg["mode"],
    prediction_type=cfg.get("prediction_type", "epsilon"),
    color_prior=cfg.get("color_prior", False),
)
net.load_state_dict(s["extra"]["ema"])
net.eval()
noise = torch.randn(1, 8, 7)
reports = []
outputs = {}
for device, threads in [("cpu", 1), ("cpu", 2), ("cuda", 1)]:
    torch.set_num_threads(threads)
    net = net.to(device)
    batch = {k: v.to(device) for k, v in b.items()}
    n = noise.to(device)
    times = []
    with torch.inference_mode():
        for i in range(60):
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            prediction = net.predict(batch) if cfg["mode"] == "act" else net.predict(batch, noise=n)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 10:
                times.append((time.perf_counter() - t) * 1000)
        outputs[device + str(threads)] = prediction.cpu().numpy()
    reports.append(
        {
            "device": device,
            "threads": threads,
            "samples": len(times),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
        }
    )
report = {
    "checkpoint": a.checkpoint,
    "batch_size": 1,
    "horizon": 8,
    "denoise_steps": 10,
    "measurements": reports,
    "cpu_cuda_max_abs_error_same_noise": float(np.abs(outputs["cpu1"] - outputs["cuda1"]).max()),
    "scope": "policy only; preprocessing/rendering excluded; warmup10; fixed noise for numerical comparison",
}
Path(a.out).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
