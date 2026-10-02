"""Direct behavioral cloning with the preserved point encoder; no oracle inputs."""
import torch
from torch import nn
from geopolicy.policies import PointCondition, Diffusion


class DirectBC(nn.Module):
    def __init__(self, history=1, horizon=8, color_prior="chroma40", binary_gripper=False):
        super().__init__()
        self.horizon, self.binary_gripper = horizon, binary_gripper
        self.condition = PointCondition(state_dim=23 * history, color_prior=color_prior)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, horizon * 7))
        self.gripper = nn.Linear(256, horizon) if binary_gripper else None

    def forward(self, batch):
        condition = self.condition(batch)
        prediction = self.head(condition).reshape(-1, self.horizon, 7)
        logits = self.gripper(condition) if self.gripper is not None else None
        return prediction, logits

    def loss(self, batch):
        prediction, logits = self(batch)
        mask = batch["action_mask"][:, :, None]
        weights = prediction.new_tensor([1, 1, 1, .1, .1, .1, 1])
        reconstruction = ((prediction - batch["action"]).square() * mask * weights).sum() / (mask.sum() * weights.sum())
        loss = reconstruction
        if logits is not None:
            target = (batch["physical_action"][:, :, 6] > 0).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = loss + (bce * mask[:, :, 0]).sum() / mask.sum()
        return loss, {"action_mse": float(reconstruction.detach())}

    def predict(self, batch):
        prediction, logits = self(batch)
        if logits is not None:
            physical = torch.where(logits >= 0, 1., -1.)
            prediction[:, :, 6] = (physical - batch["action_mean"][6]) / batch["action_std"][6]
        return prediction


def build(cfg):
    if cfg["model"] == "v1_diffusion":
        assert not cfg["binary_gripper"], "V1 recipe has its preserved continuous output"
        return Diffusion(state_dim=23 * cfg["history"], horizon=cfg["horizon"],
                         prediction_type="sample", color_prior=cfg["color_prior"])
    assert cfg["model"] == "direct_bc"
    return DirectBC(cfg["history"], cfg["horizon"], cfg["color_prior"], cfg["binary_gripper"])
