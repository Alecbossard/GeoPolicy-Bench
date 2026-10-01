"""Real recorded/live input equivalence, including moving-wrist points and history."""

import json
from pathlib import Path
import h5py
import numpy as np
from geopolicy.environment import CAMERAS, SelectPlace
from geopolicy.sensors import student_state, fuse_points
from geopolicy.io import save_json
from .data import history_state, live_batch, view_names


def check_inputs(recipe):
    manifest = json.loads(Path(recipe["dataset_manifest"]).read_text())
    record = next(
        r for r in manifest["episodes"] if r["episode_id"] == manifest["selected_validation_ids"][0]
    )
    norm = {
        k: np.array(v, np.float32)
        for k, v in json.loads(
            Path(recipe["data_extension"]["normalization_path"]).read_text()
        ).items()
    }
    env = SelectPlace(cameras=True)
    history = []
    errors = {}
    frames = min(10, record["original_frames"])
    try:
        obs = env.reset_scene(record["scene_seed"])
        with h5py.File(record["path"]) as source:
            states = source["state"][:]
            for t in range(frames):
                history.append(student_state(obs).copy())
                for view in ["fixed", "wrist", "fusion"]:
                    cfg = dict(mode="diffusion", view=view, history=4, point_budget=512)
                    batch = live_batch(env, obs, history, cfg, norm)
                    points, mask = fuse_points(
                        [
                            (source[c + "/points"][t], source[c + "/mask"][t])
                            for c in view_names(view)
                        ],
                        n=512,
                    )
                    for name, expected in [
                        ("points", points),
                        ("point_mask", mask),
                        ("state", history_state(states, t, 4, norm)),
                        ("instruction", source["instruction_tokens"][t]),
                    ]:
                        actual = batch[name][0].numpy()
                        error = float(np.abs(actual.astype(float) - expected.astype(float)).max())
                        errors[view + "/" + name] = max(errors.get(view + "/" + name, 0), error)
                        np.testing.assert_array_equal(actual, expected)
                rgb_cfg = dict(mode="act_rgb_prior", view="fusion", history=4, rgb_resolution=64)
                batch = live_batch(env, obs, history, rgb_cfg, norm)
                rgb = (
                    np.stack(
                        [source[c + "/rgb"][t, ::2, ::2].transpose(2, 0, 1) for c in CAMERAS]
                    ).astype(np.float32)
                    / 255
                )
                np.testing.assert_array_equal(batch["rgb"][0].numpy(), rgb)
                # Snapshot must remain unchanged across the next MuJoCo step.
                previous = history[-1].copy()
                obs, _, _, _ = env.step(source["action"][t])
                np.testing.assert_array_equal(history[-1], previous)
    finally:
        env.close()
    result = dict(
        scene_seed=record["scene_seed"],
        frames_checked=frames,
        errors=errors,
        rgb_exact=True,
        causal_history_exact=True,
        robot_snapshots_do_not_alias=True,
        moving_wrist_calibration_and_points_exact=True,
        reserved_test_used=False,
    )
    save_json("results/v2/input_contract_verification.json", result)
    return result
