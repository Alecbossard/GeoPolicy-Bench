"""Direct behavioral cloning with the preserved point encoder; no oracle inputs."""
import math
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


class RoutedPointCondition(PointCondition):
    """Same encoder computation, also exposes its four sensor-derived moments."""
    def forward(self, batch):
        points, valid = batch["points"], batch["point_mask"]
        features = self.mlp(points)
        b, _, width = features.shape
        query = self.query(batch["instruction"]).reshape(b, 4, width)
        logits = torch.einsum("bhc,bnc->bhn", query, features) / math.sqrt(width)
        red, green, blue = points[:, :, 3], points[:, :, 4], points[:, :, 5]
        brightness = red + green + blue + .01
        if self.color_prior == "chroma40":
            evidence = torch.stack([red-torch.maximum(green, blue), green-torch.maximum(red, blue),
                                    blue-torch.maximum(red, green), torch.minimum(red, green)-blue], 1)
            evidence = evidence / brightness[:, None] * torch.tanh(3 * brightness[:, None])
            logits = logits + 40 * evidence
        elif self.color_prior:
            evidence = torch.stack([red-green-blue, green-red-blue, blue-red-green,
                                    (red+green)/2-blue-(red-green).abs()/2], 1)
            logits = logits + 20 * evidence
        logits = logits.masked_fill(~valid[:, None], -1e4)
        weights = logits.softmax(-1) * valid[:, None]
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)
        pooled = torch.einsum("bhn,bnc->bhc", weights, features).flatten(1)
        moments = torch.einsum("bhn,bnc->bhc", weights, points)
        maximum = features.masked_fill(~valid[:, :, None], -1e4).max(1).values
        maximum = torch.where(valid.any(1)[:, None], maximum, torch.zeros_like(maximum))
        condition = self.output(torch.cat([pooled, maximum, moments.flatten(1), batch["state"], batch["instruction"]], -1))
        return condition, moments


class RoutedBC(DirectBC):
    """Label-selected sensor moments bypass the shared bottleneck; actions learned."""
    def __init__(self, history=1, horizon=8, color_prior="chroma40", binary_gripper=False, routing=True):
        super().__init__(history, horizon, color_prior, binary_gripper)
        self.condition = RoutedPointCondition(state_dim=23*history, color_prior=color_prior)
        self.head = nn.Sequential(nn.Linear(256+12, 256), nn.ReLU(), nn.Linear(256, horizon*7))
        self.routing = routing

    def forward(self, batch):
        condition, moments = self.condition(batch)
        labels = batch["instruction"]
        selected_object = (moments[:, :2] * labels[:, :2, None]).sum(1)
        selected_goal = (moments[:, 2:] * labels[:, 2:, None]).sum(1)
        route = torch.cat([selected_object, selected_goal], -1)
        if not self.routing:
            route = torch.zeros_like(route)
        prediction = self.head(torch.cat([condition, route], -1)).reshape(-1, self.horizon, 7)
        return prediction, self.gripper(condition) if self.gripper is not None else None


def build(cfg):
    if cfg["model"] == "v1_diffusion":
        assert not cfg["binary_gripper"], "V1 recipe has its preserved continuous output"
        return Diffusion(state_dim=23 * cfg["history"], horizon=cfg["horizon"],
                         prediction_type="sample", color_prior=cfg["color_prior"])
    if cfg["model"] == "routed_bc":
        return RoutedBC(cfg["history"], cfg["horizon"], cfg["color_prior"], cfg["binary_gripper"], cfg.get("routing", True))
    assert cfg["model"] == "direct_bc"
    return DirectBC(cfg["history"], cfg["horizon"], cfg["color_prior"], cfg["binary_gripper"])
