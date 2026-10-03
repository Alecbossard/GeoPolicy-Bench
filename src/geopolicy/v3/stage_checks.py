"""Replay both labels and compare recorded/live inputs on progressive tasks."""
import json
import h5py
import numpy as np
from geopolicy.sensors import camera_packet, fuse_points, student_state
from geopolicy.v2.data import view_names
from .common import ROOT, sha, write
from .stages import StageData, environment, stage_live


def stage_checks(task):
    manifest = ROOT / f"configs/v3/{task}_dataset.json"
    records = json.loads(manifest.read_text())
    selected = []
    labels = set()
    for record in records:
        label = (0,0) if task == "single" else (record["object_id"], record["goal_id"])
        if record["split"] == "train" and record["success"] and label not in labels:
            selected.append(record)
            labels.add(label)
    assert len(labels) == {"single":1, "two_objects_one_goal":2, "full":4}[task]
    cfg = dict(task=task, horizon=8, history=4, continued=True, limit=None, view="fusion",
               dataset_manifest_sha256=sha(manifest))
    # Normalize from recorded robot/action prefixes only; no privileged fields.
    states, actions = [], []
    for record in records:
        if record["split"] != "train" or not record["success"]:
            continue
        with h5py.File(ROOT / record["path"]) as f:
            states.append(f["state"][:record["original_frames"]])
            actions.append(f["action"][:record["original_frames"]])
    norm = {name + suffix: (array.mean(0) if suffix == "_mean" else np.maximum(array.std(0), .01)).astype(np.float32)
            for name, array in (("state", np.concatenate(states)), ("action", np.concatenate(actions)))
            for suffix in ("_mean", "_std")}
    env = environment(task)
    env.horizon = 240
    comparisons = []
    try:
        for record in selected:
            obs = env.reset_scene(record["scene_seed"])
            assert len(env.object_ids) == (1 if task == "single" else 2)
            assert len(env.goal_body_ids) == (2 if task == "full" else 1)
            history = []
            errors = dict(robot_state=0., normalized_history=0., points=0., instruction=0.)
            compared = 0
            with h5py.File(ROOT / record["path"]) as f:
                recorded_states = f["state"][:]
                recorded_actions = f["action"][:]
                recorded_tokens = f["instruction_tokens"][:]
                checkpoints = {0, 20, 50, 90, record["frames"] - 1}
                for t, action in enumerate(recorded_actions):
                    state = student_state(obs).copy()
                    history.append(state)
                    errors["robot_state"] = max(errors["robot_state"], float(np.max(np.abs(state - recorded_states[t]))))
                    if t in checkpoints:
                        for view in ("fixed", "wrist", "fusion"):
                            batch = stage_live(env, obs, history, dict(cfg, view=view), norm)
                            expected = np.concatenate([(recorded_states[max(0, t-lag)] - norm["state_mean"]) / norm["state_std"]
                                                       for lag in range(cfg["history"])])
                            errors["normalized_history"] = max(errors["normalized_history"], float(np.max(np.abs(expected - batch["state"][0].numpy()))))
                            points, mask = fuse_points([(f[c]["points"][t], f[c]["mask"][t]) for c in view_names(view)])
                            assert np.array_equal(mask, batch["point_mask"][0].numpy())
                            errors["points"] = max(errors["points"], float(np.max(np.abs(points - batch["points"][0].numpy()))))
                            errors["instruction"] = max(errors["instruction"], float(np.max(np.abs(recorded_tokens[t] - batch["instruction"][0].numpy()))))
                            compared += 1
                    obs, _, _, _ = env.step(action)
            assert all(value == 0 for value in errors.values()), errors
            comparisons.append(dict(scene_seed=record["scene_seed"], labels=record["instruction"],
                                    replay_steps=len(recorded_actions), three_view_comparisons=compared,
                                    maximum_errors=errors))
    finally:
        env.close()
    result = dict(task=task, dataset_manifest_sha256=cfg["dataset_manifest_sha256"],
                  comparisons=comparisons, interpretation="Recorded/live inputs only, teacher replay diagnostic; not learned-policy success or test evaluation")
    write(ROOT / f"results/v3/{task}_input_checks.json", result)
    print(json.dumps(result), flush=True)
    return result
