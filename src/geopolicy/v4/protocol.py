"""One-way reservation: checkpoint/recipe/corruption/source identities frozen."""

from datetime import datetime, timezone
import importlib.metadata
import torch
from .common import ROOT, read, plan, sha, write
from .evaluation import source_identity
from .training import tensor_hash


def freeze():
    target = ROOT / "configs/v4/final_protocol.json"
    assert not target.exists(), "Protocol already frozen"
    assert not list((ROOT / "results/v4/test").glob("*.json")), "Test already inspected"
    checks = read("results/v4/input_checks.json")
    assert checks["cuda_augmented_resume_next_update_exact"]
    decision = read("results/v4/validation_decision.json")
    assert decision["proceed_to_reserved_test"]
    registry = []
    for view in ("fixed", "fusion"):
        for aug in (False, True):
            for seed in (0, 1, 2):
                name = f'{view}_{"aug" if aug else "clean"}_s{seed}'
                cp = f"artifacts/v4/main_runs/{name}/best.pt"
                data = torch.load(ROOT / cp, map_location="cpu", weights_only=False)
                assert data["config"]["updates"] == 2000
                assert (
                    read(f"artifacts/v4/main_runs/{name}/manifest.json")[
                        "completed_update"
                    ]
                    == 2000
                )
                assert len(data["extra"]["data_identity"]) == 79
                registry.append(
                    dict(
                        name=name,
                        group=name.rsplit("_s", 1)[0],
                        training_seed=seed,
                        checkpoint=cp,
                        sha256=sha(ROOT / cp),
                        selected_update=data["progress"]["update"],
                        ema_hash=tensor_hash(data["extra"]["ema"]),
                        config=data["config"],
                    )
                )
    write(
        target,
        dict(
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            plan_sha256=sha(ROOT / "configs/v4/plan.json"),
            sources=source_identity(),
            registry=registry,
            dataset_manifest_sha256=sha(ROOT / "configs/v3/single_dataset.json"),
            validation_decision_sha256=sha(
                ROOT / "results/v4/validation_decision.json"
            ),
            packages={
                k: importlib.metadata.version(k)
                for k in ("torch", "numpy", "mujoco", "robosuite", "h5py")
            },
            reserved_test_scenes=plan()["reserved_test_scenes"],
            selection=plan()["selection"],
        ),
    )
    print("V4 protocol frozen:12checkpoints,10conditions,20newscenes", flush=True)
