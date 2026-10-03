"""V3 closed-loop evaluation with identity guards and no student action assistance."""

import csv
import hashlib
import json
import time
import numpy as np
import torch
from geopolicy.environment import CAMERAS
from geopolicy.sensors import student_state
from geopolicy.v2.resources import ResourceWatch
from .common import ROOT, sha, write
from .data import live
from .environment import SinglePlace
from .metrics import Tracker
from .models import build


def evaluate(
    name,
    checkpoint=None,
    first=110100,
    episodes=20,
    reference=False,
    raw=False,
    videos=0,
):
    assert name.replace("_", "").isalnum()
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    assert (
        first < 200000
    ), "Fresh reserved test remains locked until a V3 protocol is frozen"
    torch.set_num_threads(2)
    model, saved, cfg = None, None, {}
    if not reference:
        assert checkpoint is not None
        checkpoint = ROOT / checkpoint
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = saved["config"]
        model = build(cfg).eval()
        model.load_state_dict(saved["model"] if raw else saved["extra"]["ema"])
    task = cfg.get("task", "single")
    if task == "single":
        assert 10000 <= first < 10100 or 110000 <= first < 110300
    else:
        stage = json.loads((ROOT / "configs/v3/task_stages.json").read_text())[task]
        assert any(
            start <= first < start + 100
            for start in (
                stage["train_first"],
                stage["validation_first"],
                stage["tuning_first"],
                stage["confirmation_first"],
            )
        )
    source_paths = [
        "src/geopolicy/v3/environment.py",
        "src/geopolicy/v3/metrics.py",
        "src/geopolicy/v3/data.py",
        "src/geopolicy/v3/models.py",
        "src/geopolicy/v3/evaluation.py",
    ]
    if task != "single":
        source_paths.append("src/geopolicy/v3/stages.py")
    identity = dict(
        checkpoint_sha256=None if reference else sha(checkpoint),
        first=first,
        episodes=episodes,
        reference=reference,
        raw_weights=raw,
        plan_sha256=sha(ROOT / "configs/v3/plan.json"),
        sources={p: sha(ROOT / p) for p in source_paths},
    )
    out = ROOT / "artifacts/v3/evaluations" / name
    ip = out / "identity.json"
    if ip.exists():
        assert json.loads(ip.read_text()) == identity, "Evaluation identity changed"
    write(ip, identity)
    rp = out / "rollouts.json"
    rows = json.loads(rp.read_text()) if rp.exists() else []
    completed = {r["scene_seed"] for r in rows}
    if task == "single":
        env = SinglePlace(cameras=True)
        tracker_class, batch_function = Tracker, live
    else:
        from .stages import environment, StageTracker, stage_live

        env = environment(task)
        tracker_class, batch_function = StageTracker, stage_live
    env.horizon = plan["maximum_steps"]
    watch = ResourceWatch(out, plan["resource_limits"])
    try:
        for seed in range(first, first + episodes):
            if seed in completed:
                continue
            obs = env.reset_scene(seed)
            digest = hashlib.sha256(student_state(obs).tobytes())
            for camera in CAMERAS:
                digest.update(obs[camera + "_image"].tobytes())
                digest.update(obs[camera + "_depth"].tobytes())
            initial_hash = digest.hexdigest()
            torch.manual_seed(seed + cfg.get("seed", 0) * 1000000)
            tracker = tracker_class(plan["stability"])
            history, queue, frames = [], [], []
            started = time.perf_counter()
            for step in range(env.horizon):
                watch.sample()
                history.append(student_state(obs).copy())
                if reference:
                    action = env.reference_action()
                else:
                    if not queue:
                        batch = batch_function(
                            env, obs, history, cfg, saved["normalization"]
                        )
                        with torch.inference_mode():
                            predicted = model.predict(batch)[
                                0, : cfg["execute_steps"]
                            ].numpy()
                        norm = saved["normalization"]
                        queue = list(
                            np.clip(
                                predicted * norm["action_std"] + norm["action_mean"],
                                -1,
                                1,
                            )
                        )
                    action = queue.pop(0)
                if seed - first < videos:
                    frames.append(
                        np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1)
                    )
                obs, _, done, info = env.step(action)
                both_success = tracker.update(env, obs, action, info)
                if both_success or done:
                    break
            if frames:
                import imageio.v2 as iio

                frames.append(
                    np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1)
                )
                iio.mimwrite(out / f"scene_{seed}.mp4", frames, fps=20)
            row = dict(
                scene_seed=seed,
                training_seed=cfg.get("seed"),
                model=cfg.get("model", "privileged_reference"),
                checkpoint_sha256=identity["checkpoint_sha256"],
                diagnostic_oracle=reference,
                overfit_diagnostic=cfg.get("overfit_diagnostic", False),
                task=plan["task"] if task == "single" else task,
                steps=step + 1,
                wall_seconds=time.perf_counter() - started,
                initial_sensor_sha256=initial_hash,
                **tracker.summary(),
                collision=bool(info["collision"]),
            )
            write(out / "traces" / f"{seed}.json", tracker.trace)
            rows.append(row)
            write(rp, rows)
            write(
                ROOT / "results/v3/validation" / f"{name}.json",
                dict(identity=identity, rollouts=rows),
            )
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    cp = ROOT / "results/v3/validation" / f"{name}.csv"
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
