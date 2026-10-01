import numpy as np
import torch
from geopolicy.v2.models import build_model


def test_diffused_gripper_is_supervised_even_with_binary_deployment_head():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    cfg = dict(
        mode="diffusion",
        history=4,
        horizon=8,
        binary_gripper=True,
        color_prior="chroma40",
        diffusion_gripper_aux_weight=0.1,
    )
    norm = dict(action_mean=np.zeros(7, np.float32), action_std=np.ones(7, np.float32))
    net = build_model(cfg, norm)
    batch = dict(
        state=torch.randn(2, 92),
        instruction=torch.tensor([[1.0, 0, 1, 0]]).repeat(2, 1),
        points=torch.rand(2, 16, 6),
        point_mask=torch.ones(2, 16, dtype=torch.bool),
        action=torch.randn(2, 8, 7),
        action_mask=torch.ones(2, 8, dtype=torch.bool),
    )
    loss, metrics = net.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert net.output.weight.grad[6].abs().sum() > 0
    assert net.gripper[-1].weight.grad.abs().sum() > 0
    assert metrics["diffused_gripper_mse"] > 0
