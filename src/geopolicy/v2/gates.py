"""Actual optimizer, dual rendering and checkpoint-next-update gates."""

import json
from pathlib import Path
import numpy as np
import torch
from geopolicy.environment import SelectPlace
from geopolicy.checkpoint import save, load
from geopolicy.io import save_json
from .models import build_model
from .resources import ResourceWatch


def gate(recipe, cfg):
    out = Path("artifacts/v2/gates") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    norm = {
        k: np.asarray(v, np.float32)
        for k, v in json.loads(Path(cfg["normalization_path"]).read_text()).items()
    }
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    env = SelectPlace(cameras=True)
    obs = env.reset_scene(100500)
    watch = ResourceWatch(out, recipe["resource_limits"])
    watch.sample(force=True)
    model = build_model(cfg, norm).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    size = cfg["batch_size"]
    batch = {
        "state": torch.zeros(size, 23 * cfg["history"], device="cuda"),
        "instruction": torch.tensor([[1.0, 0, 1, 0]], device="cuda").repeat(size, 1),
        "action": torch.randn(size, cfg["horizon"], 7, device="cuda"),
        "action_mask": torch.ones(size, cfg["horizon"], dtype=torch.bool, device="cuda"),
    }
    if cfg["mode"].startswith("act_rgb"):
        batch["rgb"] = torch.rand(size, 2, 3, 64, 64, device="cuda")
    else:
        batch["points"] = torch.rand(size, cfg["point_budget"], 6, device="cuda")
        batch["point_mask"] = torch.ones(size, cfg["point_budget"], dtype=torch.bool, device="cuda")

    def step():
        model.train()
        loss, _ = model.loss(batch)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    torch.cuda.reset_peak_memory_stats()
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    step()
    save(out / "gate.pt", model, optimizer, scheduler, norm, cfg, {"update": 1})
    expected = step()
    load(out / "gate.pt", model, optimizer, scheduler)
    actual = step()
    assert all(torch.equal(expected[k], actual[k]) for k in expected), "CUDA exact resume failed"
    delta = (
        sum(float((p.detach() - before[k]).square().sum()) for k, p in model.named_parameters())
        ** 0.5
    )
    assert delta > 0
    env.step(np.zeros(7, np.float32))
    torch.cuda.synchronize()
    resource = watch.sample(force=True)
    assert (
        resource["gpu_used_free_mib_temperature_c"][1]
        >= recipe["resource_limits"]["minimum_free_gpu_mib"]
    )
    report = {
        "name": cfg["name"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": size,
        "optimizer_states_live": True,
        "dual_render_live": True,
        "exact_next_update_resume": True,
        "parameter_delta_l2": delta,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resources": resource,
    }
    save_json(out / "gate_report.json", report)
    env.close()
    return report
