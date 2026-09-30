"""Atomic, resumable optimization checkpoints. Only load trusted local files."""

import os
import random
from pathlib import Path
import numpy as np
import torch


def random_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_random(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save(path, model, optimizer, scheduler, normalization, config, progress, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "normalization": normalization,
            "config": config,
            "progress": progress,
            "random": random_state(),
            "extra": extra,
        },
        tmp,
    )
    os.replace(tmp, path)


def load(path, model, optimizer=None, scheduler=None, resume=True):
    data = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])
    if scheduler is not None and data["scheduler"] is not None:
        scheduler.load_state_dict(data["scheduler"])
    if resume:
        restore_random(data["random"])
    return data
