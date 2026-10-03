"""Diagnose the rejected state-history pilot without attributing a unique cause."""

import collections
import json
from .common import ROOT, write


def check_history():
    import h5py
    import numpy as np
    import torch
    from geopolicy.sensors import student_state
    from .data import live
    from .environment import SinglePlace
    from .models import build

    torch.set_num_threads(2)
    models = {}
    for variant in ("original", "state_history4"):
        saved = torch.load(
            ROOT / f"artifacts/v3/runs/bc_{variant}80_s0/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = build(saved["config"]).eval()
        model.load_state_dict(saved["extra"]["ema"])
        models[variant] = (model, saved)
    env = SinglePlace(cameras=True)
    errors, state_error = collections.defaultdict(list), 0.0
    try:
        obs = env.reset_scene(10000)
        history = []
        with h5py.File(ROOT / "artifacts/v3/single_dataset/episode_010000.h5") as f:
            states, actions = f["state"][:], f["action"][:]
            count = json.loads(f.attrs["metadata"])["original_frames"]
            for t in range(count):
                history.append(student_state(obs).copy())
                for variant, (model, saved) in models.items():
                    cfg, norm = saved["config"], saved["normalization"]
                    batch = live(env, obs, history, cfg, norm)
                    expected = np.concatenate(
                        [
                            (states[max(0, t - lag)] - norm["state_mean"])
                            / norm["state_std"]
                            for lag in range(cfg["history"])
                        ]
                    )
                    state_error = max(
                        state_error,
                        float(np.max(np.abs(expected - batch["state"][0].numpy()))),
                    )
                    with torch.inference_mode():
                        normalized = model.predict(batch)[0, 0].numpy()
                    physical = normalized * norm["action_std"] + norm["action_mean"]
                    errors[variant].append(
                        dict(
                            step=t,
                            phase=env.phase,
                            motion_command_mae=float(
                                np.abs(physical[:3] - actions[t, :3]).mean()
                            ),
                            gripper_sign_correct=bool(
                                (physical[6] > 0) == (actions[t, 6] > 0)
                            ),
                        )
                    )
                obs, _, _, _ = env.step(actions[t])
        assert state_error == 0, "Recorded/live causal history mismatch"
    finally:
        env.close()
    stages = {}
    for seed in (0, 1, 2):
        rows = json.loads(
            (
                ROOT / f"results/v3/validation/bc_state_history480_s{seed}_tuning.json"
            ).read_text()
        )["rollouts"]
        stages[str(seed)] = dict(collections.Counter(r["failure_stage"] for r in rows))
    post_release = summarize_history_traces()
    result = dict(
        recorded_replay_steps=count,
        maximum_history_input_error=state_error,
        order="newest to oldest, initial state repeated for missing past frames",
        closed_loop_failure_stages=stages,
        recorded_teacher_path_predictions=errors,
        post_release_motion_and_contacts=post_release,
        conclusion="No input-order or normalization mismatch found in this replay. History recipe is rejected on behavior; this does not establish a unique temporal/optimization/control cause.",
    )
    write(ROOT / "results/v3/history_diagnosis.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k != "recorded_teacher_path_predictions"
            }
        ),
        flush=True,
    )


def summarize_history_traces():
    """No renderer or Torch needed to examine the already executed rollouts."""
    cfg = json.loads((ROOT / "configs/v3/plan.json").read_text())["stability"]
    post_release = []
    for seed in (0, 1, 2):
        root = (
            ROOT / f"artifacts/v3/evaluations/bc_state_history480_s{seed}_tuning/traces"
        )
        for path in sorted(root.glob("*.json")):
            tail = [s for s in json.loads(path.read_text()) if s["release_seen"]]
            start, maximum = None, 0.0
            for s in tail:
                valid = (
                    s["inside"]
                    and cfg["center_height_m"][0]
                    < s["selected_xyz_m"][2]
                    < cfg["center_height_m"][1]
                    and s["linear_speed_m_s"] <= cfg["maximum_linear_speed_m_s"]
                    and s["angular_speed_rad_s"] <= cfg["maximum_angular_speed_rad_s"]
                )
                if not valid:
                    start = None
                elif start is None:
                    start = s["time_s"]
                if start is not None:
                    maximum = max(maximum, s["time_s"] - start)
            post_release.append(
                dict(
                    seed=seed,
                    scene_seed=int(path.stem),
                    object_quiet_dwell_s=maximum,
                    samples=len(tail),
                    finger_contact_samples=sum(s["finger_contact"] for s in tail),
                    excessive_linear_speed_samples=sum(
                        s["linear_speed_m_s"] > cfg["maximum_linear_speed_m_s"]
                        for s in tail
                    ),
                    excessive_angular_speed_samples=sum(
                        s["angular_speed_rad_s"] > cfg["maximum_angular_speed_rad_s"]
                        for s in tail
                    ),
                )
            )
    return post_release
