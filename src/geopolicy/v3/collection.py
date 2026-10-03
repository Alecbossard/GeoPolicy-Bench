"""Scripted V3 demonstrations with explicit provenance and exact prefix boundary."""

import json
import os
from pathlib import Path
import h5py
import numpy as np
from geopolicy.environment import CAMERAS
from geopolicy.sensors import camera_packet, student_state
from geopolicy.v2.resources import ResourceWatch
from .common import ROOT, sha, write
from .environment import SinglePlace
from .metrics import Tracker


def collect(first, episodes):
    assert 10000 <= first < 10100 or 110000 <= first < 110010
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    out = ROOT / "artifacts/v3/single_dataset"
    out.mkdir(parents=True, exist_ok=True)
    env = SinglePlace(cameras=True)
    env.horizon = plan["maximum_steps"]
    watch = ResourceWatch(out, plan["resource_limits"])
    try:
        for seed in range(first, first + episodes):
            path = out / f"episode_{seed:06d}.h5"
            if path.exists():
                continue
            obs = env.reset_scene(seed)
            tracker = Tracker(plan["stability"])
            frames, original_frames = [], None
            for step in range(env.horizon):
                watch.sample()
                action = env.reference_action()
                packet = camera_packet(env, obs)
                packet.update(
                    state=student_state(obs).copy(),
                    action=action.copy(),
                    instruction_tokens=np.array([1, 0, 1, 0], np.float32),
                )
                frames.append(packet)
                obs, _, done, info = env.step(action)
                tracker.update(env, obs, action, info)
                if info["success"] and original_frames is None:
                    original_frames = len(frames)
                if done or (
                    original_frames is not None and len(frames) >= original_frames + 30
                ):
                    break
            metadata = dict(
                scene_seed=seed,
                episode_id=seed,
                split="train" if seed < 100000 else "validation",
                success=tracker.physical.success,
                original_frames=original_frames,
                frames=len(frames),
                instruction=env.instruction,
                provenance="scripted_ground_truth_phase_reference_DIAGNOSTIC_teacher",
                scene_config=plan["task"],
                **tracker.summary(),
            )
            tmp = path.with_suffix(".tmp")
            with h5py.File(tmp, "w") as f:
                f.attrs["metadata"] = json.dumps(metadata)
                for key in (
                    "state",
                    "action",
                    "instruction_tokens",
                    "timestamp_s",
                    "world_from_base",
                ):
                    f.create_dataset(
                        key,
                        data=np.array([p[key] for p in frames]),
                        compression="gzip",
                        shuffle=True,
                    )
                for camera in CAMERAS:
                    group = f.create_group(camera)
                    for key in frames[0][camera]:
                        group.create_dataset(
                            key,
                            data=np.array([p[camera][key] for p in frames]),
                            compression="gzip",
                            shuffle=True,
                        )
            write(out / "teacher_traces" / f"{seed}.json", tracker.trace)
            os.replace(tmp, path)
            print(json.dumps(metadata), flush=True)
    finally:
        env.close()
    records = []
    for path in sorted(out.glob("*.h5")):
        with h5py.File(path) as f:
            records.append(
                dict(
                    json.loads(f.attrs["metadata"]),
                    path=path.relative_to(ROOT).as_posix(),
                    sha256=sha(path),
                )
            )
    write(ROOT / "configs/v3/single_dataset.json", records)
    manifest_sha = sha(ROOT / "configs/v3/single_dataset.json")
    write(ROOT / "configs/v3/datasets" / f"{manifest_sha}.json", records)
    return records
