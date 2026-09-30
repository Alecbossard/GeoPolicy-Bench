import json
from pathlib import Path
import numpy as np
import torch
from geopolicy.data import Episodes
from geopolicy.policies import make_policy
from geopolicy.checkpoint import save, load

torch.set_num_threads(2)
reports = []
for mode in ["act", "mono", "fusion"]:
    data = Episodes("artifacts/data_smoke", mode=mode)
    norm = data.normalization()
    batch = {k: v.cuda() for k, v in data.batch(np.random.default_rng(0), 4, norm).items()}
    net = make_policy(mode).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, 100, 0.99)
    torch.cuda.reset_peak_memory_stats()
    loss, metrics = net.loss(batch)
    loss.backward()
    opt.step()
    sched.step()
    path = Path("artifacts") / f"{mode}_smoke.pt"
    save(
        path,
        net,
        opt,
        sched,
        norm,
        {
            "mode": mode,
            "prediction_type": "sample" if mode != "act" else None,
            "color_prior": "chroma40" if mode != "act" else False,
        },
        {"update": 1},
    )
    load(path, net, opt, sched)
    with torch.no_grad():
        prediction = net.predict(batch)
    assert prediction.shape == (4, 8, 7) and torch.isfinite(prediction).all()
    reports.append(
        dict(
            mode=mode,
            parameters=sum(p.numel() for p in net.parameters()),
            loss=float(loss),
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            **metrics,
        )
    )
    del net, opt, batch
    torch.cuda.empty_cache()
Path("artifacts/policy_pilots.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
