"""Targeted V1 ACT, data/action, gripper and closed-loop stage measurements."""

import json
from pathlib import Path
import h5py
import numpy as np
import torch
from geopolicy.data import Episodes
from geopolicy.environment import SelectPlace, CAMERAS
from geopolicy.evaluation import observation_batch
from geopolicy.io import save_json
from .evaluation import load_model, evaluate
from .preservation import preserve_v1


def diagnose(recipe):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    report = {"preservation": preserve_v1(), "test_data_used": False}
    dataset = Episodes(recipe["dataset"], split="validation", mode="act")
    act, saved = load_model("artifacts/main_runs/act_s0/best.pt")
    norm = saved["normalization"]
    rng = np.random.default_rng(12)
    totals = {"prior": [], "posterior_oracle": []}
    for _ in range(40):
        batch = dataset.batch(rng, 32, norm)
        with torch.inference_mode():
            predictions = {
                "prior": act.predict(batch),
                "posterior_oracle": act(batch, batch["action"])[0],
            }
        target = batch["action"] * torch.from_numpy(norm["action_std"]) + torch.from_numpy(
            norm["action_mean"]
        )
        for name, pred in predictions.items():
            raw = pred * torch.from_numpy(norm["action_std"]) + torch.from_numpy(
                norm["action_mean"]
            )
            valid = batch["action_mask"]
            totals[name].append(
                {
                    "normalized_l1": float((pred - batch["action"]).abs()[valid].mean()),
                    "xyz_command_mae": float(
                        (raw[:, :, :3] - target[:, :, :3]).abs()[valid].mean()
                    ),
                    "gripper_sign_accuracy": float(
                        ((raw[:, :, 6] > 0) == (target[:, :, 6] > 0))[valid].float().mean()
                    ),
                    "first_action_gripper_sign_accuracy": float(
                        ((raw[:, 0, 6] > 0) == (target[:, 0, 6] > 0)).float().mean()
                    ),
                }
            )
    report["act_recorded_validation"] = {
        name: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for name, rows in totals.items()
    }
    report["act_note"] = (
        "Posterior uses future actions and is an offline oracle only. Zero-latent inference is standard upstream, not itself a bug."
    )
    manifest = json.loads(Path(recipe["dataset_manifest"]).read_text())
    rows = [
        r for r in manifest["episodes"] if r["episode_id"] in manifest["selected_validation_ids"]
    ]
    row = rows[0]
    env = SelectPlace(cameras=True)
    obs = env.reset_scene(row["scene_seed"])
    with h5py.File(row["path"]) as f:
        online, _ = observation_batch(env, obs, "act", norm)
        rgb = (
            np.stack([f[c + "/rgb"][0, ::2, ::2].transpose(2, 0, 1) for c in CAMERAS]).astype(
                np.float32
            )
            / 255
        )
        state = (f["state"][0] - norm["state_mean"]) / norm["state_std"]
        report["first_frame_train_inference_consistency"] = {
            "scene_seed": row["scene_seed"],
            "rgb_max_abs_error": float(np.abs(online["rgb"][0].numpy() - rgb).max()),
            "normalized_state_max_abs_error": float(
                np.abs(online["state"][0].numpy() - state).max()
            ),
        }
        actions = f["action"][:]
        roundtrip = ((actions - norm["action_mean"]) / norm["action_std"]) * norm[
            "action_std"
        ] + norm["action_mean"]
        report["action_normalization_roundtrip_max_error"] = float(
            np.abs(roundtrip - actions).max()
        )
    widths = {}
    for sign in [-1, 1]:
        obs = env.reset_scene(100500)
        initial = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        for _ in range(8):
            obs, _, _, _ = env.step(np.r_[np.zeros(6), sign])
        widths[str(sign)] = {
            "before_m": initial,
            "after_m": float(np.abs(obs["robot0_gripper_qpos"]).sum()),
        }
    env.close()
    report["physical_gripper_sign_probe"] = widths
    assert widths["-1"]["after_m"] > widths["-1"]["before_m"]
    assert widths["1"]["after_m"] < widths["1"]["before_m"]
    save_json("results/v2/diagnostics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    for name, path in [
        ("act_v1", "artifacts/main_runs/act_s0/best.pt"),
        ("mono_v1", "artifacts/main_runs/mono_s0/best.pt"),
        ("fusion_v1", "artifacts/main_runs/fusion_s0/best.pt"),
        ("reference", None),
    ]:
        evaluate(
            recipe,
            path,
            f"artifacts/v2/diagnostics/{name}",
            100200,
            20,
            videos=1,
            reference=name == "reference",
        )


def inspect_act(recipe):
    """Ablate recorded validation images/instructions; compare physical commands."""
    torch.set_num_threads(2)
    torch.manual_seed(12)
    dataset = Episodes("artifacts/dataset", split="validation", mode="act")
    model, saved = load_model("artifacts/main_runs/act_s0/best.pt")
    norm = saved["normalization"]
    totals = {
        key: []
        for key in ["zero_images", "permuted_images", "flipped_object_token", "flipped_goal_token"]
    }
    rng = np.random.default_rng(22)
    first_frames = []
    for episode in dataset.episodes:
        first_frames.append(
            {
                "state": (episode["state"][0] - norm["state_mean"]) / norm["state_std"],
                "instruction": episode["instruction_tokens"][0],
                "rgb": episode["rgb"][0].transpose(0, 3, 1, 2).astype(np.float32) / 255,
            }
        )
    batch = {k: torch.from_numpy(np.stack([r[k] for r in first_frames])) for k in first_frames[0]}
    with torch.inference_mode():
        normal = model.predict(batch)
        for name in totals:
            modified = {k: v.clone() for k, v in batch.items()}
            if name == "zero_images":
                modified["rgb"].zero_()
            elif name == "permuted_images":
                modified["rgb"] = modified["rgb"][
                    torch.from_numpy(rng.permutation(len(first_frames)))
                ]
            elif name == "flipped_object_token":
                modified["instruction"][:, :2] = modified["instruction"][:, [1, 0]]
            else:
                modified["instruction"][:, 2:] = modified["instruction"][:, [3, 2]]
            changed = model.predict(modified)
            delta = (changed - normal) * torch.from_numpy(norm["action_std"])
            totals[name] = {
                "mean_abs_xyz_command_change": float(delta[:, :, :3].abs().mean()),
                "first_xyz_command_l2_change": float(delta[:, 0, :3].norm(dim=1).mean()),
                "first_gripper_sign_changes": int(
                    (
                        (changed[:, 0, 6] * norm["action_std"][6] + norm["action_mean"][6] > 0)
                        != (normal[:, 0, 6] * norm["action_std"][6] + norm["action_mean"][6] > 0)
                    ).sum()
                ),
            }
    report = {
        "validation_first_frames": len(first_frames),
        "interventions": totals,
        "test_data_used": False,
        "interpretation": "Sensitivity is a local functional probe, not proof of correct grounding or a demonstrated cause of closed-loop failure.",
    }
    save_json("results/v2/act_input_sensitivity.json", report)
    print(json.dumps(report, indent=2))
