"""Export and replay a compact local V4 checkpoint without demonstrations."""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf8")


def export():
    import torch
    from geopolicy.v4.common import sha

    protocol = json.loads(
        (ROOT / "configs/v4/final_protocol.json").read_text(encoding="utf8")
    )
    record = next(r for r in protocol["registry"] if r["name"] == "fusion_aug_s0")
    name = "final_fusion_aug_s0_absent"
    result = json.loads(
        (ROOT / f"results/v4/test/{name}.json").read_text(encoding="utf8")
    )
    row = next(r for r in result["rollouts"] if r["scene_seed"] == 500000)
    original = torch.load(
        ROOT / record["checkpoint"], map_location="cpu", weights_only=False
    )
    out = ROOT / "artifacts/v4/demo"
    out.mkdir(parents=True, exist_ok=True)
    compact = dict(
        config=original["config"],
        normalization=original["normalization"],
        extra=dict(ema=original["extra"]["ema"]),
    )
    torch.save(compact, out / "checkpoint.pt")
    source = ROOT / f"artifacts/v4/evaluations/{name}"
    shutil.copy2(source / "initial/500000.npz", out / "expected_initial.npz")
    shutil.copy2(source / "traces/500000.json", out / "expected_trace.json")
    dump(out / "expected_row.json", row)
    manifest = dict(
        name=record["name"],
        condition="absent",
        scene_seed=500000,
        selection_rule="First reserved scene500000, fusion_aug seed0, fixed camera absent; chosen before viewing test outcomes. This is an illustration, not a new evaluation.",
        source_checkpoint=record["checkpoint"],
        source_sha256=record["sha256"],
        compact_sha256=sha(out / "checkpoint.pt"),
        source_ema_hash=record["ema_hash"],
        expected_trace_sha256=sha(out / "expected_trace.json"),
        expected_initial_sha256=sha(out / "expected_initial.npz"),
    )
    dump(out / "manifest.json", manifest)
    bundle = ROOT / "artifacts/v4/portable_demo"
    for p in sorted((ROOT / "src/geopolicy").rglob("*.py")):
        dst = bundle / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for rel in [
        "configs/v4/plan.json",
        "configs/v3/plan.json",
        "requirements-lock.txt",
        "scripts/v4/demo.py",
    ]:
        dst = bundle / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for p in out.iterdir():
        if p.is_file() and p.name in (
            "checkpoint.pt",
            "expected_initial.npz",
            "expected_trace.json",
            "expected_row.json",
            "manifest.json",
        ):
            dst = bundle / "artifacts/v4/demo" / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    (bundle / "README.md").write_text(
        "# Local V4 demo\n\nFrom this directory, use pinned Python3.11 dependencies and run:\n\n```powershell\npython scripts/v4/demo.py replay\n```\n\nNo training data required. One cube, one tray, fixed instruction; fixed RGB-D camera representation absent. Source scene500000 was predeclared, not chosen by success. Raw simulator RGB and the effective sampled-point input are both shown. No real-robot transfer claim.\n",
        encoding="utf8",
    )
    with zipfile.ZipFile(
        ROOT / "artifacts/v4/demo_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(bundle).as_posix())
    dump(
        ROOT / "results/v4/demo_export.json",
        dict(
            **manifest,
            bundle_sha256=sha(ROOT / "artifacts/v4/demo_bundle.zip"),
            bundle_bytes=(ROOT / "artifacts/v4/demo_bundle.zip").stat().st_size,
            compact_bytes=(out / "checkpoint.pt").stat().st_size,
        ),
    )


def replay(output):
    import numpy as np
    import torch
    import imageio.v2 as iio
    from PIL import Image, ImageDraw
    from geopolicy.v3.environment import SinglePlace
    from geopolicy.v3.models import build
    from geopolicy.v3.metrics import Tracker
    from geopolicy.sensors import camera_packet, student_state
    from geopolicy.v4.perturbations import CAMERAS, perturb, evaluation_rng
    from geopolicy.v4.data import live_batch
    from geopolicy.v4.common import read, sha
    from geopolicy.v2.resources import ResourceWatch
    from geopolicy.v4.training import tensor_hash

    output = (ROOT / output).resolve()
    assert output.is_relative_to(ROOT / "artifacts/v4")
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "artifacts/v4/demo"
    meta = read(base / "manifest.json")
    assert sha(base / "checkpoint.pt") == meta["compact_sha256"]
    assert sha(base / "expected_trace.json") == meta["expected_trace_sha256"]
    saved = torch.load(base / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = saved["config"]
    norm = saved["normalization"]
    assert tensor_hash(saved["extra"]["ema"]) == meta["source_ema_hash"]
    torch.set_num_threads(2)
    net = build(cfg).eval()
    net.load_state_dict(saved["extra"]["ema"])
    expected = read(base / "expected_trace.json")
    row = read(base / "expected_row.json")
    plan = read("configs/v4/plan.json")
    condition = plan["conditions"][meta["condition"]]
    env = SinglePlace(cameras=True)
    env.horizon = 240
    tracker = Tracker(read("configs/v3/plan.json")["stability"])
    watch = ResourceWatch(output, plan["resource_limits"])
    frames = []
    history = []
    queue = []
    sensor_exact = True
    actions_exact = True
    physics_exact = True
    try:
        obs = env.reset_scene(meta["scene_seed"])
        initial = camera_packet(env, obs)
        degraded = perturb(initial, condition, evaluation_rng(meta["scene_seed"], 0))
        with np.load(base / "expected_initial.npz") as f:
            sensor_exact &= np.array_equal(f["state"], student_state(obs))
            for c in CAMERAS:
                for k in (
                    "rgb",
                    "depth_m",
                    "intrinsic",
                    "base_from_camera",
                    "points",
                    "mask",
                ):
                    sensor_exact &= np.array_equal(
                        initial[c][k], f[c + "__clean__" + k]
                    )
                for k in ("points", "mask"):
                    sensor_exact &= np.array_equal(
                        degraded[c][k], f[c + "__corrupted__" + k]
                    )
        for step, target in enumerate(expected):
            watch.sample()
            history.append(student_state(obs).copy())
            if not queue:
                packet = initial if step == 0 else camera_packet(env, obs)
                degraded = perturb(
                    packet, condition, evaluation_rng(meta["scene_seed"], step)
                )
                with torch.inference_mode():
                    prediction = net.predict(live_batch(degraded, history, cfg, norm))[
                        0, : cfg["execute_steps"]
                    ].numpy()
                queue = list(
                    np.clip(
                        prediction * norm["action_std"] + norm["action_mean"], -1, 1
                    )
                )
            # Top row shows physical simulator RGB. Bottom shows the actual retained
            # point representation; fixed-camera absence is visible as a blank panel.
            rgb = np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1)
            effective = []
            for c in CAMERAS:
                pic = np.zeros((128, 128, 3), np.uint8)
                q = degraded[c]
                m = q["mask"]
                xyz = (q["points"][m, :3] - q["base_from_camera"][:3, 3]) @ q[
                    "base_from_camera"
                ][:3, :3]
                if len(xyz):
                    z = xyz[:, 2]
                    u = np.rint(
                        xyz[:, 0] / z * q["intrinsic"][0, 0] + q["intrinsic"][0, 2]
                    ).astype(int)
                    v = np.rint(
                        xyz[:, 1] / z * q["intrinsic"][1, 1] + q["intrinsic"][1, 2]
                    ).astype(int)
                    valid = (u >= 0) & (u < 128) & (v >= 0) & (v < 128)
                    pic[v[valid], u[valid]] = np.clip(
                        q["points"][m, 3:][valid] * 255, 0, 255
                    ).astype(np.uint8)
                effective.append(pic)
            image = Image.fromarray(
                np.concatenate([rgb, np.concatenate(effective, 1)], 0)
            ).resize((512, 512), Image.Resampling.NEAREST)
            banner = Image.new("RGB", (512, 560), (25, 25, 25))
            banner.paste(image, (0, 48))
            draw = ImageDraw.Draw(banner)
            draw.text(
                (8, 5),
                "V4 fusion+aug | fixed camera absent | scene500000",
                fill="white",
            )
            draw.text(
                (8, 22),
                "Top: simulator RGB / Bottom: effective XYZRGB input",
                fill="white",
            )
            frames.append(np.asarray(banner))
            action = queue.pop(0)
            actions_exact &= np.array_equal(
                action, np.asarray(target["action"], np.float32)
            )
            obs, _, done, info = env.step(action)
            tracker.update(env, obs, action, info)
            actual = tracker.trace[-1]
            for k in (
                "selected_xyz_m",
                "rotation",
                "goal_xyz_m",
                "finger_width_m",
                "linear_speed_m_s",
                "angular_speed_rad_s",
            ):
                physics_exact &= np.array_equal(
                    np.asarray(actual[k]), np.asarray(target[k])
                )
    finally:
        env.close()
    assert len(frames) == len(expected)
    summary = tracker.summary()
    summary_exact = all(summary[k] == row[k] for k in summary)
    iio.mimwrite(output / "demo.mp4", frames, fps=20)
    iio.mimsave(
        output / "demo.gif",
        [Image.fromarray(x).resize((384, 420)) for x in frames[::4]],
        duration=0.2,
        loop=0,
    )
    dump(output / "trace.json", tracker.trace)
    dump(output / "rollout.json", summary)
    verified = dict(
        initial_modalities_exact=bool(sensor_exact),
        actions_exact=bool(actions_exact),
        physics_exact=bool(physics_exact),
        summary_exact=bool(summary_exact),
        frames=len(frames),
        compact_sha256=sha(base / "checkpoint.pt"),
        video_sha256=sha(output / "demo.mp4"),
        rule=meta["selection_rule"],
        source=meta["source_checkpoint"],
    )
    dump(output / "verification.json", verified)
    print(json.dumps(verified), flush=True)
    assert (
        sensor_exact and actions_exact and physics_exact and summary_exact
    ), "Replay mismatch preserved in verification.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["export", "replay"])
    p.add_argument("--output", default="artifacts/v4/demo/reproduced")
    args = p.parse_args()
    from geopolicy.v4.resources import preflight

    preflight("demo " + args.command)
    if args.command == "export":
        export()
    else:
        replay(args.output)


if __name__ == "__main__":
    main()
