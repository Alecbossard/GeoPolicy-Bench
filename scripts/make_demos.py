"""Replay earliest observed success/failure and caption actual simulator RGB frames."""

import json
import subprocess
import sys
from pathlib import Path
import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

rows = json.loads(Path("artifacts/final_evaluations/fusion_s0/nominal/rollouts.json").read_text())
out = Path("results/demos")
out.mkdir(parents=True, exist_ok=True)
fontpath = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
title = ImageFont.truetype(str(fontpath), 22)
body = ImageFont.truetype(str(fontpath), 17)
small = ImageFont.truetype(str(fontpath), 14)
manifest = []
for wanted, label in [(True, "success"), (False, "failure")]:
    row = next(r for r in sorted(rows, key=lambda r: r["scene_seed"]) if r["success"] == wanted)
    root = Path("artifacts/demo_replays") / label
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.log").open("w") as log:
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "geopolicy.cli",
                "evaluate",
                "--checkpoint",
                row["checkpoint"],
                "--out",
                str(root),
                "--episodes",
                "1",
                "--first-seed",
                str(row["scene_seed"]),
                "--conditions",
                "nominal",
                "--device",
                "cpu",
                "--execute-steps",
                "2",
                "--videos",
                "1",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert code == 0
    replay = json.loads((root / "rollouts.json").read_text())[0]
    assert replay["success"] == row["success"] and replay["steps"] == row["steps"]
    assert replay["first_action"] == row["first_action"]
    source = next(root.glob("*.mp4"))
    reader = iio.get_reader(source)
    frames = []
    for index, frame in enumerate(reader):
        canvas = Image.new("RGB", (768, 544), (16, 24, 36))
        draw = ImageDraw.Draw(canvas)
        color = (94, 210, 175) if wanted else (240, 167, 81)
        draw.text(
            (12, 10), f"{label.upper()} | COMPACT 3D DIFFUSION / FUSION", font=title, fill=color
        )
        draw.text((12, 42), row["instruction"], font=body, fill="white")
        draw.text((12, 73), "Fixed RGB view", font=small, fill=(190, 202, 218))
        draw.text((396, 73), "Wrist RGB view", font=small, fill=(190, 202, 218))
        image = Image.fromarray(frame[:, :, :3]).resize((768, 384), Image.Resampling.NEAREST)
        canvas.paste(image, (0, 96))
        draw.text(
            (12, 489),
            f"Test scene {row['scene_seed']} | training seed0 | sim t={index/20:.2f}s",
            font=body,
            fill="white",
        )
        draw.text(
            (12, 518),
            "Simulation only | RGB-D128 | control20Hz | horizon8 / execute2",
            font=small,
            fill=(190, 202, 218),
        )
        frames.append(np.asarray(canvas))
    reader.close()
    iio.mimwrite(out / f"{label}.mp4", frames, fps=20)
    sampled = frames[::4]
    if not np.array_equal(sampled[-1], frames[-1]):
        sampled.append(frames[-1])
    iio.mimsave(out / f"{label}.gif", sampled, duration=200, loop=0)
    iio.imwrite(out / f"{label}_final.png", frames[-1])
    manifest.append(
        {
            "outcome": label,
            "selection": "earliest matching scene in completed fusion_s0 nominal100 test cell",
            "scene_seed": row["scene_seed"],
            "instruction": row["instruction"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "video": str(out / f"{label}.mp4"),
            "gif": str(out / f"{label}.gif"),
            "source_video": str(source),
            "replay_success_and_step_and_first_action_exact": True,
            "terminal_placement_error_difference_m": abs(
                replay["placement_error_m"] - row["placement_error_m"]
            ),
            "frames": len(frames),
            "simulation_seconds": row["simulation_seconds"],
            "no_generated_or_interpolated_physics_frames": True,
        }
    )
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
