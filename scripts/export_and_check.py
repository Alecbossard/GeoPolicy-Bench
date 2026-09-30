"""Export real trained denoiser, compare full DDIM and warmed CPU timings."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from geopolicy.exporting import export_denoiser, install_onnx_denoiser
from geopolicy.data import Episodes
from geopolicy.policies import make_policy

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="artifacts/main_runs/fusion_s0/best.pt")
p.add_argument("--out", default="artifacts/onnx_export")
a = p.parse_args()
torch.set_num_threads(2)
torch.manual_seed(42)
report = export_denoiser(a.checkpoint, a.out)
saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
cfg = saved["config"]
model = make_policy(
    cfg["mode"], prediction_type=cfg["prediction_type"], color_prior=cfg["color_prior"]
)
model.load_state_dict(saved["extra"]["ema"])
model.eval()
data = Episodes("artifacts/dataset", "validation", cfg["mode"], limit=1)
batch = data.batch(np.random.default_rng(42), 1, saved["normalization"])
noise = torch.randn(1, 8, 7)
measurements = {}
outputs = {}
for backend in ["pytorch", "onnx_denoiser"]:
    if backend == "onnx_denoiser":
        install_onnx_denoiser(model, Path(a.out) / "denoiser.onnx")
    times = []
    with torch.inference_mode():
        for i in range(60):
            start = time.perf_counter()
            output = model.predict(batch, noise=noise)
            if i >= 10:
                times.append((time.perf_counter() - start) * 1000)
        outputs[backend] = output.numpy()
    measurements[backend] = {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "batch_size": 1,
        "samples": 50,
        "warmup": 10,
    }
report["full_ddim_normalized_action_max_abs_error"] = float(
    np.abs(outputs["pytorch"] - outputs["onnx_denoiser"]).max()
)
report["numeric_pass"] &= report["full_ddim_normalized_action_max_abs_error"] < 1e-4
report["full_policy_cpu2_measurements"] = measurements
(Path(a.out) / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report["numeric_pass"]
