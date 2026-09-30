import numpy as np
import random
import torch
from geopolicy.checkpoint import save, load


def test_resume_reproduces_next_optimizer_step_and_rng(tmp_path):
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95)

    def step():
        x = torch.randn(4, 3)
        loss = net(x).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        return [p.detach().clone() for p in net.parameters()], np.random.rand(), random.random()

    step()
    save(tmp_path / "checkpoint.pt", net, opt, sched, {"mean": [0]}, {"seed": 11}, {"update": 1})
    expected, a, b = step()
    data = load(tmp_path / "checkpoint.pt", net, opt, sched)
    actual, c, d = step()
    assert data["progress"]["update"] == 1
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert a == c and b == d


def test_real_diffusion_optimizer_resume_is_exact(tmp_path):
    from geopolicy.policies import Diffusion

    torch.set_num_threads(2)
    torch.manual_seed(17)
    net = Diffusion()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 10)
    batch = {
        "state": torch.zeros(2, 23),
        "instruction": torch.tensor([[1.0, 0, 1, 0]]).repeat(2, 1),
        "points": torch.rand(2, 32, 6),
        "point_mask": torch.ones(2, 32, dtype=torch.bool),
        "action": torch.randn(2, 8, 7),
        "action_mask": torch.ones(2, 8, dtype=torch.bool),
    }

    def step():
        loss, _ = net.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        return [p.detach().clone() for p in net.parameters()]

    step()
    save(tmp_path / "diffusion.pt", net, opt, scheduler, {}, {"mode": "fusion"}, {"update": 1})
    expected = step()
    load(tmp_path / "diffusion.pt", net, opt, scheduler)
    actual = step()
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
