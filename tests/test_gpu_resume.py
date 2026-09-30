"""Opt-in CUDA exact-next-update gate; default suite stays CPU only."""

import os
import pytest
from geopolicy.runtime import setup

setup()
import torch
from geopolicy.policies import Diffusion
from geopolicy.checkpoint import save, load


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("GEO_GPU_TESTS") != "1", reason="Set GEO_GPU_TESTS=1 explicitly for CUDA gate"
)
def test_cuda_diffusion_resume_reproduces_next_update(tmp_path):
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = Diffusion().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    batch = {
        "state": torch.zeros(4, 23, device="cuda"),
        "instruction": torch.tensor([[1.0, 0, 1, 0]], device="cuda").repeat(4, 1),
        "points": torch.rand(4, 512, 6, device="cuda"),
        "point_mask": torch.ones(4, 512, dtype=torch.bool, device="cuda"),
        "action": torch.randn(4, 8, 7, device="cuda"),
        "action_mask": torch.ones(4, 8, dtype=torch.bool, device="cuda"),
    }

    def step():
        loss, _ = model.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return [p.detach().clone() for p in model.parameters()]

    step()
    save(tmp_path / "cuda.pt", model, optimizer, scheduler, {}, {"mode": "fusion"}, {"update": 1})
    expected = step()
    load(tmp_path / "cuda.pt", model, optimizer, scheduler)
    actual = step()
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
