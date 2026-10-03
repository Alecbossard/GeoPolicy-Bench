"""Checkpoint-only replay; compares actions/physics with a frozen test episode."""

import argparse
import hashlib
import json
import importlib.metadata
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    from geopolicy.v3.resources import preflight

    preflight("compact_demo_replay")
    import numpy as np
    import torch
    from geopolicy.environment import CAMERAS
    from geopolicy.sensors import student_state
    from geopolicy.v3.common import sha, write
    from geopolicy.v3.final_protocol import json_hash, source_hash, tensor_hash
    from geopolicy.v3.environment import SinglePlace
    from geopolicy.v3.models import build
    from geopolicy.v3.data import live
    from geopolicy.v3.metrics import Tracker
    from geopolicy.v2.resources import ResourceWatch

    torch.set_num_threads(2)
    home = ROOT if (ROOT / "checkpoint.pt").exists() else ROOT / "artifacts/v3/demo"
    proof = json.loads((home / "equivalence.json").read_text())
    protocol = json.loads((ROOT / "configs/v3/final_protocol.json").read_text())
    assert list(sys.version_info[:3]) == protocol["python_version"]
    assert all(
        importlib.metadata.version(name) == version
        for name, version in protocol["packages"].items()
    )
    record = next(r for r in protocol["registry"] if r["key"] == proof["key"])
    assert proof["source_checkpoint_sha256"] == record["checkpoint_sha256"]
    assert sha(home / "checkpoint.pt") == proof["compact_sha256"]
    for name, expected in protocol["sources"].items():
        assert source_hash(ROOT / name) == expected, name
    saved = torch.load(home / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["config"] == record["training_config"]
    assert tensor_hash(saved["extra"]["ema"]) == record["ema_tensor_sha256"]
    assert (
        json_hash({k: v.tolist() for k, v in saved["normalization"].items()})
        == record["normalization_sha256"]
    )
    model = build(saved["config"]).eval()
    model.load_state_dict(saved["extra"]["ema"])
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    assert json_hash(plan) == protocol["plan_sha256"]
    scene = proof["scene_seed"]
    assert scene == protocol["first"]
    cfg, norm = saved["config"], saved["normalization"]
    output = Path(args.output).resolve() if args.output else home / "replay"
    assert (
        output.is_relative_to((ROOT / "artifacts/v3").resolve())
        if home != ROOT
        else output.is_relative_to(ROOT)
    )
    output.mkdir(parents=True, exist_ok=True)
    env = SinglePlace(cameras=True)
    env.horizon = plan["maximum_steps"]
    watch = ResourceWatch(output, plan["resource_limits"])
    frames, history, queue = [], [], []
    tracker = Tracker(plan["stability"])
    try:
        obs = env.reset_scene(scene)
        digest = hashlib.sha256(student_state(obs).tobytes())
        for camera in CAMERAS:
            digest.update(obs[camera + "_image"].tobytes())
            digest.update(obs[camera + "_depth"].tobytes())
        torch.manual_seed(scene + cfg["seed"] * 1000000)
        for step in range(env.horizon):
            watch.sample()
            history.append(student_state(obs).copy())
            if not queue:
                batch = live(env, obs, history, cfg, norm)
                assert set(batch) <= {
                    "state",
                    "instruction",
                    "points",
                    "point_mask",
                    "action_mean",
                    "action_std",
                }
                with torch.inference_mode():
                    prediction = model.predict(batch)[0, : cfg["execute_steps"]].numpy()
                queue = list(
                    np.clip(
                        prediction * norm["action_std"] + norm["action_mean"], -1, 1
                    )
                )
            frames.append(np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1))
            action = queue.pop(0)
            obs, _, done, info = env.step(action)
            if tracker.update(env, obs, action, info) or done:
                break
        frames.append(np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1))
    finally:
        env.close()
    assert digest.hexdigest() == proof["initial_sensor_sha256"]
    assert (
        json_hash(tracker.trace) == proof["frozen_trace_sha256"]
    ), "Compact replay differs from frozen test actions/physics"
    assert tracker.summary() == proof["frozen_summary"]
    import imageio.v2 as iio

    iio.mimwrite(output / "demo.mp4", frames, fps=20)
    iio.imwrite(output / "first_frame.png", frames[0])
    iio.imwrite(output / "last_frame.png", frames[-1])
    (output / "trace.json").write_text(
        json.dumps(tracker.trace, indent=2), encoding="utf8"
    )
    result = dict(
        scene_seed=scene,
        compact_sha256=proof["compact_sha256"],
        exact_frozen_trace=True,
        imported_source_directory=str(ROOT / "src/geopolicy"),
        policy_device="cpu",
        runtime_packages=protocol["packages"],
        **tracker.summary(),
    )
    (output / "verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
