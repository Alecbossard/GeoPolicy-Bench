"""Focused ACT-style and diffusion controls; no privileged inputs or pretrained weights."""

import torch
from torch import nn
from torch.nn import functional as F
from geopolicy.policies import ACT, Diffusion, PointCondition


class ActionLoss:
    def install_normalization(self, norm):
        self.register_buffer("action_mean", torch.as_tensor(norm["action_mean"]).clone())
        self.register_buffer("action_std", torch.as_tensor(norm["action_std"]).clone())

    def control_loss(self, prediction, logit, batch):
        mask = batch["action_mask"]
        weights = prediction.new_tensor([1, 1, 1, 0.1, 0.1, 0.1])
        motion = (
            (prediction[..., :6] - batch["action"][..., :6]).square() * weights * mask[:, :, None]
        ).sum() / (mask.sum() * weights.sum())
        physical = batch["action"][..., 6] * self.action_std[6] + self.action_mean[6]
        grip = (
            F.binary_cross_entropy_with_logits(logit, (physical > 0).float(), reduction="none")
            * mask
        ).sum() / mask.sum()
        return motion + grip, {
            "motion_mse": float(motion.detach()),
            "gripper_bce": float(grip.detach()),
        }

    def normalized_gripper(self, logit):
        physical = torch.where(logit >= 0, torch.ones_like(logit), -torch.ones_like(logit))
        return (physical - self.action_mean[6]) / self.action_std[6]


class DiffusionControl(Diffusion, ActionLoss):
    def __init__(self, cfg, norm):
        super().__init__(
            state_dim=23 * cfg["history"], horizon=cfg["horizon"], color_prior=cfg["color_prior"]
        )
        self.binary_gripper = cfg["binary_gripper"]
        self.install_normalization(norm)
        if self.binary_gripper:
            self.gripper = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, cfg["horizon"])
            )

    def loss(self, batch):
        if not self.binary_gripper:
            return super().loss(batch)
        target = batch["action"]
        t = torch.randint(self.timesteps, (len(target),), device=target.device)
        a = self.alpha_bar[t][:, None, None]
        condition = self.condition(batch)
        noisy = a.sqrt() * target + (1 - a).sqrt() * torch.randn_like(target)
        prediction = self.denoise(noisy, t, condition)
        return self.control_loss(prediction, self.gripper(condition), batch)

    def predict(self, batch, inference_steps=10, noise=None):
        prediction = super().predict(batch, inference_steps, noise)
        if self.binary_gripper:
            prediction[..., 6] = self.normalized_gripper(self.gripper(self.condition(batch)))
        return prediction


class ACTPrior(ACT, ActionLoss):
    """Train the exact zero-latent deployment path; unused CVAE stays frozen."""

    def __init__(self, cfg, norm):
        super().__init__(state_dim=23 * cfg["history"], horizon=cfg["horizon"])
        self.binary_gripper = cfg["binary_gripper"]
        self.install_normalization(norm)
        for module in [self.vae, self.stats, self.actions]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.cls.requires_grad = False
        self.vae_pos.requires_grad = False

    def loss(self, batch):
        prediction = self(batch)[0]
        if self.binary_gripper:
            return self.control_loss(prediction, prediction[..., 6], batch)
        mask = batch["action_mask"][:, :, None]
        loss = ((prediction - batch["action"]).abs() * mask).sum() / (mask.sum() * 7)
        return loss, {"prior_l1": float(loss.detach())}

    def predict(self, batch):
        prediction = self(batch)[0]
        if self.binary_gripper:
            prediction[..., 6] = self.normalized_gripper(prediction[..., 6].clone())
        return prediction


class ACTPoint(nn.Module, ActionLoss):
    """Deterministic action-chunk Transformer with explicit 3D input adaptation."""

    def __init__(self, cfg, norm):
        super().__init__()
        self.condition = PointCondition(
            state_dim=23 * cfg["history"], color_prior=cfg["color_prior"]
        )
        self.projection = nn.Linear(256, 128)
        self.queries = nn.Parameter(torch.randn(1, cfg["horizon"], 128) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(128, 4, 512, dropout=0, batch_first=True), 2
        )
        self.out = nn.Linear(128, 7)
        self.install_normalization(norm)

    def forward(self, batch):
        memory = self.projection(self.condition(batch))[:, None]
        return self.out(self.decoder(self.queries.expand(len(memory), -1, -1), memory))

    def loss(self, batch):
        prediction = self(batch)
        return self.control_loss(prediction, prediction[..., 6], batch)

    def predict(self, batch):
        prediction = self(batch)
        prediction[..., 6] = self.normalized_gripper(prediction[..., 6].clone())
        return prediction


def build_model(cfg, norm):
    if cfg["mode"] == "diffusion":
        return DiffusionControl(cfg, norm)
    if cfg["mode"] == "act_point":
        return ACTPoint(cfg, norm)
    if cfg["mode"].startswith("act_rgb"):
        return ACTPrior(cfg, norm)
    raise ValueError(cfg["mode"])
