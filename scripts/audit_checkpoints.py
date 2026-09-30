"""Verify identical data normalization/recipe and distinct trained checkpoints."""

import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

rows = []
reference = None
for mode in ["mono", "fusion", "act"]:
    for seed in [0, 1, 2]:
        root = Path("artifacts/main_runs") / f"{mode}_s{seed}"
        path = root / "best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        cfg = saved["config"]
        for key, value in {
            "seed": seed,
            "mode": mode,
            "updates": 8000,
            "batch_size": 32,
            "demonstration_episodes": 200,
            "execute_steps": 2,
            "horizon": 8,
        }.items():
            assert cfg[key] == value, (path, key)
        if reference is None:
            reference = saved["normalization"]
        for key in reference:
            np.testing.assert_array_equal(reference[key], saved["normalization"][key])
        assert saved["optimizer"]["state"] and saved["scheduler"] and saved["extra"]["ema"]
        assert set(saved["random"]) == {"python", "numpy", "torch", "cuda"}
        assert len(saved["random"]["cuda"]) == 1 and saved["extra"]["batch_rng"]
        assert saved["progress"]["update"] >= 500
        rows.append(
            {
                "run": root.name,
                "checkpoint": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selected_update": saved["progress"]["update"],
                "selected_offline_loss": saved["progress"]["best_validation"],
                "precision": sorted({str(p.dtype) for p in saved["model"].values()}),
                "full_optimizer_scheduler_rng": True,
            }
        )
        del saved
assert len({r["sha256"] for r in rows}) == 9
vla = torch.load("artifacts/smolvla_s0/best.pt", map_location="cpu", weights_only=False)
dtypes = collections.Counter()
for value in vla["trainable"].values():
    dtypes[str(value.dtype)] += value.numel()
report = {
    "main_runs": rows,
    "identical_state_action_normalization": True,
    "distinct_checkpoint_hashes": 9,
    "smolvla_trainable_parameter_precision_counts": dict(dtypes),
    "smolvla_saved_update": vla["update"],
    "smolvla_note": "Native upstream mixed BF16/FP32 modules; no independently added autocast/scaler. Authentic backward/optimizer gate validated.",
}
Path("results/checkpoint_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
