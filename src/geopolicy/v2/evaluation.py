"""Common sustained-placement evaluator for preserved V1 and new V2 policies."""

import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from geopolicy.environment import SelectPlace, CAMERAS
from geopolicy.evaluation import observation_batch
from geopolicy.policies import make_policy
from geopolicy.io import save_json
from geopolicy.sensors import student_state
from .config import file_hash, json_hash, source_hash
from .metrics import StageTracker
from .resources import ResourceWatch


def load_model(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    if cfg.get("version") == 2:
        from .models import build_model

        model = build_model(cfg, saved["normalization"])
    else:
        model = make_policy(
            cfg["mode"],
            prediction_type=cfg.get("prediction_type", "epsilon"),
            color_prior=cfg.get("color_prior", False),
        )
    model.load_state_dict(
        saved["extra"]["ema"] if saved.get("extra", {}).get("ema") is not None else saved["model"]
    )
    return model.to(device).eval(), saved


def evaluate(
    recipe,
    checkpoint,
    out,
    first_seed,
    episodes,
    conditions=("nominal",),
    videos=0,
    reference=False,
    object_id=None,
    goal_id=None,
    frozen=False,
    demo_record=None,
):
    torch.set_num_threads(recipe["evaluation"]["torch_threads"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if first_seed >= 300000:
        freeze = Path("configs/v2/final_protocol.json")
        assert frozen and freeze.exists(), "Reserved test is locked until protocol freeze"
        protocol = json.loads(freeze.read_text())
        assert protocol["recipe_sha256"] == json_hash(recipe), "Recipe changed after test freeze"
        checkpoint_hash = file_hash(checkpoint)
        if checkpoint_hash not in protocol["registered_checkpoint_hashes"]:
            assert demo_record is not None, "Unregistered checkpoint"
            equivalence = json.loads(Path("configs/v2/demo_equivalence.json").read_text())
            assert equivalence["compact_sha256"] == checkpoint_hash
            assert equivalence["source_sha256"] in protocol["registered_checkpoint_hashes"]
            assert equivalence["ema_tensor_exact"]
        for path, expected in protocol["evaluation_source_hashes"].items():
            assert source_hash(path) == expected, f"Evaluation source changed after freeze: {path}"
        if demo_record is not None:
            assert episodes == 1 and first_seed == demo_record["scene_seed"]
            assert list(conditions) == [demo_record["condition"]]
            assert file_hash(checkpoint) == demo_record["checkpoint_sha256"]
        elif object_id is None:
            assert first_seed == recipe["evaluation"]["reserved_test_first_seed"]
            assert episodes == recipe["evaluation"]["test_episodes"]
        else:
            assert first_seed == recipe["evaluation"]["counterfactual_first_seed"]
            assert episodes == recipe["evaluation"]["counterfactual_scenes"]
    model, saved = (
        (None, None) if reference else load_model(checkpoint, recipe["evaluation"]["device"])
    )
    cfg = {} if reference else saved["config"]
    identity = {
        "checkpoint_sha256": None if reference else file_hash(checkpoint),
        "recipe_sha256": json_hash(recipe),
        "first_seed": first_seed,
        "episodes": episodes,
        "conditions": list(conditions),
        "reference": reference,
        "object_id": object_id,
        "goal_id": goal_id,
    }
    ip = out / "identity.json"
    if ip.exists():
        assert json.loads(ip.read_text()) == identity, "Evaluation identity mismatch"
    save_json(ip, identity)
    rows = (
        json.loads((out / "rollouts.json").read_text()) if (out / "rollouts.json").exists() else []
    )
    completed = {(r["condition"], r["scene_seed"]) for r in rows}
    env = SelectPlace(cameras=True)
    env.horizon = recipe["evaluation"]["maximum_steps"]
    watch = ResourceWatch(out, recipe["resource_limits"])
    try:
        for condition in conditions:
            begun = time.monotonic()
            for scene in range(first_seed, first_seed + episodes):
                if (condition, scene) in completed:
                    continue
                obs = env.reset_scene(scene, object_id, goal_id)
                digest = hashlib.sha256(student_state(obs).tobytes())
                for camera in CAMERAS:
                    digest.update(obs[camera + "_image"].tobytes())
                    digest.update(obs[camera + "_depth"].tobytes())
                digest.update(env.sim.data.cam_xpos.tobytes())
                digest.update(env.sim.data.cam_xmat.tobytes())
                initial_sensor_sha = digest.hexdigest()
                rng = np.random.default_rng(scene)
                torch.manual_seed(scene + cfg.get("seed", 0) * 1000000)
                tracker = StageTracker(recipe["stability"])
                history = []
                queue = []
                frames = []
                policy_times = []
                pre_times = []
                start = time.perf_counter()
                for step in range(env.horizon):
                    watch.sample()
                    if time.monotonic() - begun > recipe["resource_limits"]["cell_timeout_seconds"]:
                        raise TimeoutError("Evaluation cell timeout; completed rows are preserved")
                    history.append(student_state(obs).copy())
                    if reference:
                        action = env.reference_action()
                    else:
                        if not queue:
                            tick = time.perf_counter()
                            if cfg.get("version") == 2:
                                from .data import live_batch

                                batch = live_batch(
                                    env, obs, history, cfg, saved["normalization"], condition, rng
                                )
                            else:
                                batch, _ = observation_batch(
                                    env, obs, cfg["mode"], saved["normalization"], condition, rng
                                )
                            pre_times.append(time.perf_counter() - tick)
                            tick = time.perf_counter()
                            with torch.inference_mode():
                                prediction = model.predict(batch)[
                                    0, : cfg.get("execute_steps", 2)
                                ].numpy()
                            policy_times.append(time.perf_counter() - tick)
                            norm = saved["normalization"]
                            queue = list(
                                np.clip(
                                    prediction * norm["action_std"] + norm["action_mean"], -1, 1
                                )
                            )
                        action = queue.pop(0)
                    if scene - first_seed < videos:
                        frames.append(
                            np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], axis=1)
                        )
                    if step == 0:
                        first_action = np.asarray(action).tolist()
                    obs, _, done, info = env.step(action)
                    stable = tracker.update(env, obs, action, info)
                    if stable or done:
                        break
                if frames:
                    frames.append(
                        np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], axis=1)
                    )
                    import imageio.v2 as iio

                    iio.mimwrite(
                        out
                        / f"{condition}_{scene}_{'stable' if tracker.window.success else 'failure'}.mp4",
                        frames,
                        fps=20,
                    )
                record = {
                    "scene_seed": scene,
                    "condition": condition,
                    "training_seed": cfg.get("seed"),
                    "instruction": env.instruction,
                    "object_id": env.target_object,
                    "goal_id": env.target_goal,
                    "steps": step + 1,
                    "wall_seconds": time.perf_counter() - start,
                    "simulation_seconds": (step + 1) / 20,
                    "checkpoint_sha256": identity["checkpoint_sha256"],
                    "initial_rgbd_robot_camera_pose_sha256": initial_sensor_sha,
                    "first_action": first_action,
                    "mode": cfg.get("mode", "reference"),
                    **info,
                    **tracker.summary(),
                }
                record["success"] = record["stable_success"]
                for name, values in [("policy", policy_times), ("preprocess", pre_times)]:
                    for q in [50, 95]:
                        record[f"{name}_p{q}_ms"] = (
                            float(np.percentile(values, q) * 1000) if values else 0
                        )
                for q in [50, 95]:
                    record[f"total_p{q}_ms"] = (
                        float(
                            np.percentile(np.asarray(policy_times) + np.asarray(pre_times), q)
                            * 1000
                        )
                        if policy_times
                        else 0
                    )
                save_json(out / "traces" / f"{condition}_{scene}.json", tracker.trace)
                rows.append(record)
                save_json(out / "rollouts.json", rows)
                with (out / "rollouts.csv").open("w", newline="", encoding="utf8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(record), flush=True)
    finally:
        env.close()
    return rows
