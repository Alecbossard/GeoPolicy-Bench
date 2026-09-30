"""Compact adaptations of ACT and DP3-inspired conditional action diffusion.

No pretrained backbone. Independent implementation, not exact paper reproduction.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class ACT(nn.Module):
    def __init__(self, state_dim=23, horizon=8, width=128, latent_dim=16):
        super().__init__()
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, width, 3, 2, 1),
            nn.ReLU(),
        )
        self.image_pos = nn.Parameter(torch.randn(1, 128, width) * 0.02)
        self.state = nn.Linear(state_dim, width)
        self.language = nn.Linear(4, width)
        self.latent = nn.Linear(latent_dim, width)
        self.actions = nn.Linear(7, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.vae_pos = nn.Parameter(torch.randn(1, horizon + 2, width) * 0.02)
        self.vae = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 512, dropout=0, batch_first=True), 2
        )
        self.stats = nn.Linear(width, latent_dim * 2)
        self.queries = nn.Parameter(torch.randn(1, horizon, width) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(width, 4, 512, dropout=0, batch_first=True), 2
        )
        self.out = nn.Linear(width, 7)

    def forward(self, batch, targets=None):
        state = self.state(batch["state"])
        b = len(state)
        kl = state.sum() * 0
        if targets is not None:
            seq = (
                torch.cat([self.cls.expand(b, -1, -1), state[:, None], self.actions(targets)], 1)
                + self.vae_pos
            )
            pad = torch.cat(
                [torch.zeros(b, 2, device=state.device, dtype=torch.bool), ~batch["action_mask"]], 1
            )
            enc = self.vae(seq, src_key_padding_mask=pad)[:, 0]
            mu, logvar = self.stats(enc).chunk(2, -1)
            logvar = logvar.clamp(-10, 10)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        else:
            z = state.new_zeros(b, self.latent_dim)
        rgb = batch["rgb"].reshape(b * 2, 3, 64, 64)
        image = self.cnn(rgb).flatten(2).transpose(1, 2).reshape(b, 128, -1) + self.image_pos
        memory = torch.cat(
            [
                state[:, None],
                self.language(batch["instruction"])[:, None],
                self.latent(z)[:, None],
                image,
            ],
            1,
        )
        prediction = self.out(self.decoder(self.queries.expand(b, -1, -1), memory))
        return prediction, kl

    def loss(self, batch):
        prediction, kl = self(batch, batch["action"])
        mask = batch["action_mask"][:, :, None]
        reconstruction = ((prediction - batch["action"]).abs() * mask).sum() / (mask.sum() * 7)
        loss = reconstruction + kl
        return loss, {"reconstruction": float(reconstruction.detach()), "kl": float(kl.detach())}

    def predict(self, batch):
        return self(batch)[0]


class PointCondition(nn.Module):
    def __init__(self, state_dim=23, width=128, condition_dim=256, color_prior=True):
        super().__init__()
        self.color_prior = color_prior
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.query = nn.Linear(4, width * 4)
        self.output = nn.Sequential(
            nn.Linear(width * 5 + state_dim + 4 + 4 * 6, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, batch):
        p = batch["points"]
        valid = batch["point_mask"]
        f = self.mlp(p)
        b, _, w = f.shape
        q = self.query(batch["instruction"]).reshape(b, 4, w)
        logits = torch.einsum("bhc,bnc->bhn", q, f) / math.sqrt(w)
        # Color evidence comes only from RGB, never simulator segmentation.
        # A declared fixed color prior makes rare colored surfaces accessible;
        # learned attention remains trainable on top of this sensor feature.
        red, green, blue = p[:, :, 3], p[:, :, 4], p[:, :, 5]
        brightness = red + green + blue + 0.01
        color_evidence = torch.stack(
            [
                red - torch.maximum(green, blue),
                green - torch.maximum(red, blue),
                blue - torch.maximum(red, green),
                torch.minimum(red, green) - blue,
            ],
            1,
        )
        # Normalized chromatic contrast is an RGB-only inductive prior, not a
        # guarantee of color grounding. Versioned for older pilot checkpoints.
        if self.color_prior == "chroma40":
            color_evidence = (
                color_evidence / brightness[:, None] * torch.tanh(3 * brightness[:, None])
            )
            logits = logits + 40 * color_evidence
        elif self.color_prior:
            legacy = torch.stack(
                [
                    red - green - blue,
                    green - red - blue,
                    blue - red - green,
                    (red + green) / 2 - blue - (red - green).abs() / 2,
                ],
                1,
            )
            logits = logits + 20 * legacy
        logits = logits.masked_fill(~valid[:, None], -1e4)
        weights = logits.softmax(-1) * valid[:, None]
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)
        pooled = torch.einsum("bhn,bnc->bhc", weights, f).flatten(1)
        # Learned soft attention also retains explicit XYZRGB moments for precision.
        moments = torch.einsum("bhn,bnc->bhc", weights, p).flatten(1)
        maximum = f.masked_fill(~valid[:, :, None], -1e4).max(1).values
        maximum = torch.where(valid.any(1)[:, None], maximum, torch.zeros_like(maximum))
        return self.output(
            torch.cat([pooled, maximum, moments, batch["state"], batch["instruction"]], -1)
        )


def time_embedding(t, width=64):
    periods = torch.exp(
        torch.arange(width // 2, device=t.device) * (-math.log(10000) / (width // 2 - 1))
    )
    phase = t.float()[:, None] * periods[None]
    return torch.cat([phase.sin(), phase.cos()], -1)


class FiLMBlock(nn.Module):
    def __init__(self, width, condition):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.film = nn.Linear(condition, width * 2)

    def forward(self, x, c):
        scale, shift = self.film(c).chunk(2, -1)
        z = F.silu(self.norm1(self.conv1(x)))
        z = z * (1 + scale[:, :, None]) + shift[:, :, None]
        return x + self.conv2(F.silu(self.norm2(z)))


class Diffusion(nn.Module):
    def __init__(
        self,
        state_dim=23,
        horizon=8,
        width=128,
        timesteps=100,
        prediction_type="sample",
        color_prior="chroma40",
    ):
        super().__init__()
        self.horizon = horizon
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.condition = PointCondition(state_dim, color_prior=color_prior)
        self.time = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv1d(7, width, 1)
        self.blocks = nn.ModuleList([FiLMBlock(width, 384) for _ in range(4)])
        self.output = nn.Conv1d(width, 7, 1)
        # Cosine DDPM alpha-bar, clipped betas as in improved DDPM schedules.
        grid = torch.linspace(0, timesteps, timesteps + 1)
        alpha = torch.cos(((grid / timesteps + 0.008) / 1.008) * math.pi / 2).square()
        alpha = alpha / alpha[0]
        beta = (1 - alpha[1:] / alpha[:-1]).clamp(0.0001, 0.999)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))

    def denoise(self, noisy, t, condition):
        c = torch.cat([condition, self.time(time_embedding(t))], -1)
        x = self.input(noisy.transpose(1, 2))
        for block in self.blocks:
            x = block(x, c)
        return self.output(x).transpose(1, 2)

    def loss(self, batch):
        target = batch["action"]
        b = len(target)
        t = torch.randint(self.timesteps, (b,), device=target.device)
        a = self.alpha_bar[t][:, None, None]
        noise = torch.randn_like(target)
        noisy = a.sqrt() * target + (1 - a).sqrt() * noise
        prediction = self.denoise(noisy, t, self.condition(batch))
        mask = batch["action_mask"][:, :, None]
        target_prediction = noise if self.prediction_type == "epsilon" else target
        weights = target.new_tensor([1, 1, 1, 0.1, 0.1, 0.1, 1])
        loss = ((prediction - target_prediction).square() * mask * weights).sum() / (
            mask.sum() * weights.sum()
        )
        return loss, {self.prediction_type + "_mse": float(loss.detach())}

    def predict(self, batch, inference_steps=10, noise=None):
        condition = self.condition(batch)
        b = len(condition)
        x = torch.randn(b, self.horizon, 7, device=condition.device) if noise is None else noise
        schedule = torch.linspace(self.timesteps - 1, 0, inference_steps, device=x.device).long()
        for i, step in enumerate(schedule):
            t = step.expand(b)
            a = self.alpha_bar[step]
            prev = self.alpha_bar[schedule[i + 1]] if i + 1 < len(schedule) else x.new_tensor(1.0)
            prediction = self.denoise(x, t, condition)
            if self.prediction_type == "epsilon":
                epsilon = prediction
                x0 = (x - (1 - a).sqrt() * epsilon) / a.sqrt()
            else:
                x0 = prediction
                epsilon = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x0 = x0.clamp(-5, 5)
            x = prev.sqrt() * x0 + (1 - prev).sqrt() * epsilon
        return x


def make_policy(mode, **kwargs):
    if mode == "act":
        kwargs.pop("prediction_type", None)
        kwargs.pop("color_prior", None)
    return ACT(**kwargs) if mode == "act" else Diffusion(**kwargs)
