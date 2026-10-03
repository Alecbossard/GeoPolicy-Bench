"""Paired scene corruptions; saved initial arrays and raw full physics traces."""

import hashlib
import json
import time
import numpy as np
import torch
from geopolicy.sensors import camera_packet, student_state
from geopolicy.v3.environment import SinglePlace
from geopolicy.v3.metrics import Tracker
from geopolicy.v3.models import build
from geopolicy.v2.resources import ResourceWatch
from .common import ROOT, plan, read, sha, write, runtime_identity, verify_runtime
from .perturbations import CAMERAS, perturb, packet_digest, evaluation_rng
from .data import live_batch


def source_identity():
    return {
        p.relative_to(ROOT)
        .as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest()
        for p in sorted((ROOT / "src/geopolicy").rglob("*.py"))
    }


def evaluate(
    name, checkpoint, condition, first, episodes, split="validation", videos=0
):
    assert name.replace("_", "").isalnum()
    p = plan()
    assert condition in p["conditions"]
    if split == "test":
        protocol = read("configs/v4/final_protocol.json")
        verify_runtime(protocol)
        assert protocol["sources"] == source_identity() and protocol[
            "plan_sha256"
        ] == sha(ROOT / "configs/v4/plan.json")
        registry = {r["checkpoint"]: r for r in protocol["registry"]}
        assert checkpoint in registry
        assert sha(ROOT / checkpoint) == registry[checkpoint]["sha256"]
        assert first == p["reserved_test_scenes"][0] and episodes == 20
    else:
        assert 210000 <= first <= 210109 and first + episodes - 1 <= 210109
    torch.set_num_threads(2)
    saved = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    norm = saved["normalization"]
    assert cfg["history"] == 4 and cfg["model"] == "direct_bc" and cfg["continued"]
    model = build(cfg).eval()
    model.load_state_dict(saved["extra"]["ema"])
    group = (
        cfg.get("view", "fusion") + "_" + ("aug" if cfg.get("augmented") else "clean")
    )
    identity = dict(
        checkpoint=checkpoint,
        checkpoint_sha256=sha(ROOT / checkpoint),
        condition=condition,
        perturbation=p["conditions"][condition],
        first=first,
        episodes=episodes,
        split=split,
        sources=source_identity(),
        plan_sha256=sha(ROOT / "configs/v4/plan.json"),
        **runtime_identity(),
    )
    out = ROOT / "artifacts/v4/evaluations" / name
    if (out / "identity.json").exists():
        assert read(out / "identity.json") == identity, "Evaluation identity changed"
    write(out / "identity.json", identity)
    rows = read(out / "rollouts.json") if (out / "rollouts.json").exists() else []
    completed = {r["scene_seed"] for r in rows}
    env = SinglePlace(cameras=True)
    env.horizon = 240
    stability = read("configs/v3/plan.json")["stability"]
    watch = ResourceWatch(out, p["resource_limits"])
    started_job = time.perf_counter()
    try:
        for scene in range(first, first + episodes):
            if scene in completed:
                continue
            obs = env.reset_scene(scene)
            initial = camera_packet(env, obs)
            corrupted = perturb(
                initial, p["conditions"][condition], evaluation_rng(scene, 0)
            )
            state = student_state(obs)
            clean_hash = hashlib.sha256(
                (
                    packet_digest(
                        initial,
                        (
                            "rgb",
                            "depth_m",
                            "intrinsic",
                            "base_from_camera",
                            "points",
                            "mask",
                        ),
                    )
                    + hashlib.sha256(state.tobytes()).hexdigest()
                ).encode()
            ).hexdigest()
            corrupted_hash = packet_digest(corrupted)
            arrays = {"state": state, "world_from_base": initial["world_from_base"]}
            for c in CAMERAS:
                for k in (
                    "rgb",
                    "depth_m",
                    "intrinsic",
                    "base_from_camera",
                    "points",
                    "mask",
                ):
                    arrays[c + "__clean__" + k] = initial[c][k]
                for k in ("points", "mask"):
                    arrays[c + "__corrupted__" + k] = corrupted[c][k]
            destination = out / "initial" / f"{scene}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **arrays)
            tracker = Tracker(stability)
            history = []
            queue = []
            frames = []
            coverage = []
            latencies = []
            start = time.perf_counter()
            for step in range(240):
                watch.sample()
                history.append(student_state(obs).copy())
                if not queue:
                    tick = time.perf_counter()
                    packet = initial if step == 0 else camera_packet(env, obs)
                    degraded = (
                        corrupted
                        if step == 0
                        else perturb(
                            packet,
                            p["conditions"][condition],
                            evaluation_rng(scene, step),
                        )
                    )
                    b = live_batch(degraded, history, cfg, norm)
                    coverage.append(
                        dict(
                            step=step,
                            clean={c: int(packet[c]["mask"].sum()) for c in CAMERAS},
                            corrupted={
                                c: int(degraded[c]["mask"].sum()) for c in CAMERAS
                            },
                            fused=int(b["point_mask"].sum()),
                        )
                    )
                    with torch.inference_mode():
                        predicted = model.predict(b)[0, : cfg["execute_steps"]].numpy()
                    assert np.isfinite(predicted).all()
                    queue = list(
                        np.clip(
                            predicted * norm["action_std"] + norm["action_mean"], -1, 1
                        )
                    )
                    latencies.append(time.perf_counter() - tick)
                action = queue.pop(0)
                if scene - first < videos:
                    frames.append(
                        np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1)
                    )
                obs, _, done, info = env.step(action)
                if tracker.update(env, obs, action, info) or done:
                    break
            row = dict(
                scene_seed=scene,
                training_seed=cfg["seed"],
                group=group,
                condition=condition,
                checkpoint_sha256=identity["checkpoint_sha256"],
                steps=step + 1,
                wall_seconds=time.perf_counter() - start,
                initial_clean_hash=clean_hash,
                initial_corrupted_hash=corrupted_hash,
                initial_npz_sha256=sha(destination),
                p95_packet_prediction_ms=float(np.percentile(latencies, 95) * 1000),
                **tracker.summary(),
                collision=bool(info["collision"]),
            )
            write(out / "traces" / f"{scene}.json", tracker.trace)
            write(out / "coverage" / f"{scene}.json", coverage)
            if frames:
                import imageio.v2 as iio

                iio.mimwrite(out / f"scene_{scene}.mp4", frames, fps=20)
            rows.append(row)
            write(out / "rollouts.json", rows)
            write(
                ROOT / "results/v4" / split / f"{name}.json",
                dict(identity=identity, rollouts=rows),
            )
            print(json.dumps(row), flush=True)
            if (
                time.perf_counter() - started_job
                > p["resource_limits"]["cell_timeout_seconds"]
            ):
                raise RuntimeError("Evaluation cell timeout; progress saved")
    finally:
        env.close()
    return rows
