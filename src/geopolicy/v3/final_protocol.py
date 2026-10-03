"""Prospectively frozen, resumable single-task test. No test-driven selection."""

import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from .common import ROOT, sha, write


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def source_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(
            json.dumps(
                [name, list(array.shape), str(array.dtype)], separators=(",", ":")
            ).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def specs():
    records = []
    for view in ("fixed", "wrist", "fusion"):
        for prior in (True, False):
            for seed in (0, 1, 2):
                key = f"single_{view}_{'prior' if prior else 'no_prior'}_s{seed}"
                run = (
                    f"bc_continued_history480_s{seed}"
                    if view == "fusion" and prior
                    else key
                )
                records.append(
                    dict(
                        key=key,
                        group=f"v3_{view}_{'prior' if prior else 'no_prior'}",
                        checkpoint=f"artifacts/v3/runs/{run}/best.pt",
                        seed=seed,
                        comparison="matched V3 continued/history4 view-prior study",
                    )
                )
    for group, run in (
        ("v3_original_bc", "bc_original80"),
        ("v3_v1_recipe", "diffusion_original80"),
    ):
        for seed in (0, 1, 2):
            records.append(
                dict(
                    key=f"{group}_s{seed}",
                    group=group,
                    checkpoint=f"artifacts/v3/runs/{run}_s{seed}/best.pt",
                    seed=seed,
                    comparison="matched 80-demo/2000-update V3 simplified-task reference",
                )
            )
    for version, folder in (
        (1, "artifacts/main_runs/fusion_s"),
        (2, "artifacts/v2/main_runs/fusion_prior_s"),
    ):
        for seed in (0, 1, 2):
            records.append(
                dict(
                    key=f"preserved_v{version}_fusion_s{seed}",
                    group=f"preserved_v{version}_fusion",
                    checkpoint=f"{folder}{seed}/best.pt",
                    seed=seed,
                    comparison="historical full-task model transferred to the same simplified test; different training data/budget/domain, not a controlled algorithm comparison",
                )
            )
    return records


def freeze():
    import torch
    from .pipeline import competence, validation_counts

    target = ROOT / "configs/v3/final_protocol.json"
    assert (
        not target.exists()
    ), "Protocol already frozen; never refreeze after test access"
    assert not list(
        (ROOT / "results/v3/test").glob("*.json")
    ), "Test results already exist"
    competence("bc")
    selection = json.loads((ROOT / "configs/v3/selection.json").read_text())
    assert (
        selection["recipe"] == "continued_history4"
        and not selection["reserved_test_used"]
    )
    registry = specs()
    shared_norm = None
    for record in registry:
        path = ROOT / record["checkpoint"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        cfg = saved["config"]
        assert cfg["seed"] == record["seed"]
        assert not cfg.get("overfit_diagnostic", False)
        record.update(
            checkpoint_sha256=sha(path),
            training_config=cfg,
            chosen_update=saved.get("progress", {}).get("update"),
            ema_tensor_sha256=tensor_hash(
                saved["extra"]["ema"]
                if saved.get("extra", {}).get("ema") is not None
                else saved["model"]
            ),
            normalization_sha256=json_hash(
                {k: v.tolist() for k, v in saved["normalization"].items()}
            ),
        )
        if record["group"].startswith("v3_"):
            assert cfg["version"] == 3 and cfg["updates"] == 2000 and cfg["limit"] == 80
            assert cfg.get("task", "single") == "single"
            if shared_norm is None:
                shared_norm = record["normalization_sha256"]
            assert (
                record["normalization_sha256"] == shared_norm
            ), "Normalization differs across matched V3 models"
            name = (
                record["key"] + "_tuning"
                if record["key"].startswith("single_")
                else path.parent.name + "_tuning"
            )
            result = json.loads(
                (ROOT / f"results/v3/validation/{name}.json").read_text()
            )
            validation_counts(result, record["seed"], 110100)
            assert (
                result["identity"]["checkpoint_sha256"] == record["checkpoint_sha256"]
            )
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    protocol = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        task="single",
        first=400000,
        episodes=50,
        seeds=[0, 1, 2],
        conditions=["nominal"],
        registry=registry,
        plan_sha256=json_hash(plan),
        runtime_limits_sha256=json_hash(
            json.loads((ROOT / "configs/v3/runtime_limits.json").read_text())
        ),
        sources={
            p.relative_to(ROOT).as_posix(): source_hash(p)
            for p in sorted((ROOT / "src/geopolicy").rglob("*.py"))
        },
        python=sys.version,
        python_version=list(sys.version_info[:3]),
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "mujoco", "robosuite", "h5py")
        },
        selection="Already confirmed fusion/prior/continued/history4; no further selection from test",
        limits="Fixed instruction and simplified scene; no V3 multi-object competence. Historical V1/V2 transfer is a task-matched reference with unmatched training.",
    )
    write(target, protocol)
    print(
        json.dumps(
            dict(
                frozen=True,
                checkpoints=len(registry),
                scenes=[400000, 400049],
                reserved_test_previously_accessed=False,
            )
        ),
        flush=True,
    )
    return protocol


def registered(key):
    protocol = json.loads((ROOT / "configs/v3/final_protocol.json").read_text())
    assert (
        list(sys.version_info[:3]) == protocol["python_version"]
    ), "Python version differs from frozen runtime"
    assert all(
        importlib.metadata.version(name) == version
        for name, version in protocol["packages"].items()
    ), "Frozen dependency version differs"
    matches = [record for record in protocol["registry"] if record["key"] == key]
    assert len(matches) == 1, "Unregistered final-test model"
    for relative, expected in protocol["sources"].items():
        assert (
            source_hash(ROOT / relative) == expected
        ), f"Frozen evaluation source changed: {relative}"
    assert (
        json_hash(json.loads((ROOT / "configs/v3/plan.json").read_text()))
        == protocol["plan_sha256"]
    )
    assert (
        json_hash(json.loads((ROOT / "configs/v3/runtime_limits.json").read_text()))
        == protocol["runtime_limits_sha256"]
    )
    record = matches[0]
    assert sha(ROOT / record["checkpoint"]) == record["checkpoint_sha256"]
    return protocol, record


def evaluate_final(key):
    import time
    import csv
    import numpy as np
    import torch
    from geopolicy.environment import CAMERAS
    from geopolicy.sensors import student_state
    from geopolicy.v2.evaluation import load_model
    from geopolicy.v2.resources import ResourceWatch
    from .data import live
    from .environment import SinglePlace
    from .metrics import Tracker
    from .models import build

    torch.set_num_threads(2)
    protocol, record = registered(key)
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    saved = torch.load(
        ROOT / record["checkpoint"], map_location="cpu", weights_only=False
    )
    cfg, norm = saved["config"], saved["normalization"]
    if cfg.get("version") == 3:
        model = build(cfg).eval()
        model.load_state_dict(saved["extra"]["ema"])
    else:
        model, saved = load_model(ROOT / record["checkpoint"])
    name = "final_" + key
    out = ROOT / "artifacts/v3/evaluations" / name
    identity = dict(
        checkpoint_sha256=record["checkpoint_sha256"],
        protocol_sha256=json_hash(protocol),
        first=protocol["first"],
        episodes=protocol["episodes"],
        reference=False,
        raw_weights=False,
    )
    ip = out / "identity.json"
    if ip.exists():
        assert json.loads(ip.read_text()) == identity
    write(ip, identity)
    rows = (
        json.loads((out / "rollouts.json").read_text())
        if (out / "rollouts.json").exists()
        else []
    )
    completed = {r["scene_seed"] for r in rows}
    env = SinglePlace(cameras=True)
    env.horizon = plan["maximum_steps"]
    watch = ResourceWatch(out, plan["resource_limits"])
    begun = time.monotonic()
    try:
        for scene in range(protocol["first"], protocol["first"] + protocol["episodes"]):
            if scene in completed:
                continue
            obs = env.reset_scene(scene)
            digest = hashlib.sha256(student_state(obs).tobytes())
            for camera in CAMERAS:
                digest.update(obs[camera + "_image"].tobytes())
                digest.update(obs[camera + "_depth"].tobytes())
            torch.manual_seed(scene + cfg["seed"] * 1000000)
            rng = np.random.default_rng(scene)
            history, queue, frames, policy_times, pre_times = [], [], [], [], []
            tracker = Tracker(plan["stability"])
            started = time.perf_counter()
            for step in range(env.horizon):
                watch.sample()
                if (
                    time.monotonic() - begun
                    > plan["resource_limits"]["cell_timeout_seconds"]
                ):
                    raise TimeoutError(
                        "Final cell timeout; completed episodes preserved"
                    )
                history.append(student_state(obs).copy())
                if not queue:
                    tick = time.perf_counter()
                    if cfg.get("version") == 3:
                        batch = live(env, obs, history, cfg, norm)
                    elif cfg.get("version") == 2:
                        from geopolicy.v2.data import live_batch

                        batch = live_batch(env, obs, history, cfg, norm, "nominal", rng)
                    else:
                        from geopolicy.evaluation import observation_batch

                        batch, _ = observation_batch(
                            env, obs, cfg["mode"], norm, "nominal", rng
                        )
                    pre_times.append(time.perf_counter() - tick)
                    tick = time.perf_counter()
                    with torch.inference_mode():
                        prediction = model.predict(batch)[
                            0, : cfg["execute_steps"]
                        ].numpy()
                    policy_times.append(time.perf_counter() - tick)
                    queue = list(
                        np.clip(
                            prediction * norm["action_std"] + norm["action_mean"], -1, 1
                        )
                    )
                if scene == protocol["first"] and key in (
                    "single_fusion_prior_s0",
                    "v3_original_bc_s2",
                    "v3_v1_recipe_s0",
                ):
                    frames.append(
                        np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1)
                    )
                action = queue.pop(0)
                obs, _, done, info = env.step(action)
                if tracker.update(env, obs, action, info) or done:
                    break
            if frames:
                import imageio.v2 as iio

                frames.append(
                    np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1)
                )
                iio.mimwrite(out / f"scene_{scene}.mp4", frames, fps=20)
            row = dict(
                scene_seed=scene,
                training_seed=cfg["seed"],
                group=record["group"],
                task="single",
                model=cfg.get("model", cfg.get("mode")),
                checkpoint_sha256=record["checkpoint_sha256"],
                diagnostic_oracle=False,
                overfit_diagnostic=False,
                condition="nominal",
                steps=step + 1,
                wall_seconds=time.perf_counter() - started,
                initial_sensor_sha256=digest.hexdigest(),
                preprocess_p95_ms=float(np.percentile(pre_times, 95) * 1000),
                policy_p95_ms=float(np.percentile(policy_times, 95) * 1000),
                collision=bool(info["collision"]),
                **tracker.summary(),
            )
            write(out / "traces" / f"{scene}.json", tracker.trace)
            rows.append(row)
            write(out / "rollouts.json", rows)
            write(
                ROOT / "results/v3/test" / f"{name}.json",
                dict(identity=identity, rollouts=rows),
            )
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    cp = ROOT / "results/v3/test" / f"{name}.csv"
    with cp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def final_study():
    from .pipeline import supervise

    protocol = json.loads((ROOT / "configs/v3/final_protocol.json").read_text())
    supervise(
        "frozen_final_test",
        [
            (record["key"], ["final-evaluate", "--key", record["key"]])
            for record in protocol["registry"]
        ],
    )
