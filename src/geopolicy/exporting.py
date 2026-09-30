"""Useful diffusion denoiser export, explicit scope and numerical checks."""

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .policies import make_policy


class DenoiserGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noisy_action, timestep, condition):
        return self.model.denoise(noisy_action, timestep, condition)


def install_onnx_denoiser(model, path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    def denoise(noisy_action, timestep, condition):
        inputs = {
            "noisy_action": noisy_action.detach().cpu().numpy(),
            "timestep": timestep.detach().cpu().numpy(),
            "condition": condition.detach().cpu().numpy(),
        }
        return torch.from_numpy(session.run(None, inputs)[0]).to(noisy_action.device)

    model.denoise = denoise
    return session


def export_denoiser(checkpoint, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["config"]["mode"] == "act":
        raise ValueError("This export targets compact diffusion denoiser only")
    model = make_policy(
        saved["config"]["mode"],
        prediction_type=saved["config"].get("prediction_type", "epsilon"),
        color_prior=saved["config"].get("color_prior", False),
    )
    model.load_state_dict(saved["extra"]["ema"])
    model.eval()
    graph = DenoiserGraph(model)
    args = (torch.randn(1, 8, 7), torch.tensor([50], dtype=torch.long), torch.randn(1, 256))
    path = out / "denoiser.onnx"
    torch.onnx.export(
        graph,
        args,
        path,
        input_names=["noisy_action", "timestep", "condition"],
        output_names=["prediction"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "noisy_action": {0: "batch"},
            "timestep": {0: "batch"},
            "condition": {0: "batch"},
            "prediction": {0: "batch"},
        },
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errors = []
    for t in [0, 10, 50, 99]:
        noisy = torch.randn(1, 8, 7)
        timestep = torch.tensor([t])
        condition = torch.randn(1, 256)
        with torch.inference_mode():
            expected = graph(noisy, timestep, condition).numpy()
        actual = session.run(
            None,
            {
                "noisy_action": noisy.numpy(),
                "timestep": timestep.numpy(),
                "condition": condition.numpy(),
            },
        )[0]
        errors.append(float(np.max(np.abs(expected - actual))))
    report = {
        "scope": "FP32 temporal denoiser only; point encoder, preprocessing and DDIM loop stay in PyTorch",
        "checkpoint": str(checkpoint),
        "onnx": str(path),
        "opset": 17,
        "providers": session.get_providers(),
        "max_abs_errors": errors,
        "max_abs_error": max(errors),
        "numeric_pass": max(errors) < 1e-4,
        "closed_loop_verified": False,
        "tensorrt_verified": False,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
