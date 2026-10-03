"""One-way reservation: checkpoint/recipe/corruption/source identities frozen."""

from datetime import datetime, timezone
import torch
from .common import ROOT, read, plan, sha, write, runtime_identity
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
    sources = source_identity()
    changed = [
        rel
        for rel, h in decision["confirmation_sources"].items()
        if sources.get(rel) != h
    ]
    allowed_guard_changes = {
        "src/geopolicy/v4/common.py",
        "src/geopolicy/v4/evaluation.py",
        "src/geopolicy/v4/protocol.py",
    }
    assert (
        set(changed) <= allowed_guard_changes
    ), "Scientific source changed after validation"
    assert sources.keys() == decision["confirmation_sources"].keys()
    runtime_checks = read("results/v4/io_runtime_checks.json")
    assert runtime_checks["verified"]
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
            sources=sources,
            post_validation_guard_changes=changed,
            guard_change_reason="Bounded Windows atomic-write retry and frozen Python/package verification; numerical model/data/perturbation code unchanged",
            io_runtime_checks_sha256=sha(ROOT / "results/v4/io_runtime_checks.json"),
            registry=registry,
            dataset_manifest_sha256=sha(ROOT / "configs/v3/single_dataset.json"),
            validation_decision_sha256=sha(
                ROOT / "results/v4/validation_decision.json"
            ),
            **runtime_identity(),
            reserved_test_scenes=plan()["reserved_test_scenes"],
            selection=plan()["selection"],
        ),
    )
    print("V4 protocol frozen:12checkpoints,10conditions,20newscenes", flush=True)
