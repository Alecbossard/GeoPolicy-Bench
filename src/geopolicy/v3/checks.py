"""Recorded/live alignment, physical controller and real CUDA optimizer gates."""
import copy
import json
import h5py
import imageio.v2 as iio
import numpy as np
import torch
from geopolicy.checkpoint import save, load
from geopolicy.environment import CAMERAS
from geopolicy.sensors import camera_packet, fuse_points, student_state
from geopolicy.v2.resources import ResourceWatch
from .common import ROOT, write
from .data import Data, live
from .environment import SinglePlace
from .models import build


def checks():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    out = ROOT / "artifacts/v3/checks"
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(model="direct_bc", horizon=8, history=1, binary_gripper=False,
               color_prior="chroma40", continued=True, limit=4)
    data = Data(cfg, "train")
    norm = data.normalization()
    watch = ResourceWatch(out, plan["resource_limits"])
    resource_before = watch.sample(force=True)
    env = SinglePlace(cameras=True)
    report = dict(resource_before=resource_before, teacher_demonstrations=6,
                  oracle_provenance="scripted_reference; no learned performance claim")
    try:
        assert len(env.object_ids) == len(env.goal_body_ids) == 1
        names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
        assert "receptacle_1" not in names and "distractor" not in names
        assert env.other.root_body not in names
        report["one_physical_cube_and_tray"] = True
        path = ROOT / "artifacts/v3/single_dataset/episode_010000.h5"
        history = []
        maximum = dict(state=0., points=0., rgb=0., action_normalization_roundtrip=0.)
        obs = env.reset_scene(10000)
        with h5py.File(path) as f:
            for t in range(len(f["action"])):
                history.append(student_state(obs).copy())
                maximum["state"] = max(maximum["state"], float(np.max(np.abs(history[-1] - f["state"][t]))))
                if t in (0, 20, 50, 90):
                    batch = live(env, obs, history, cfg, norm)
                    maximum["points"] = max(maximum["points"], float(np.max(np.abs(batch["points"][0].numpy() - data.episodes[0]["points"][t]))))
                    for c in CAMERAS:
                        maximum["rgb"] = max(maximum["rgb"], float(np.max(np.abs(obs[c + "_image"][::-1].astype(int) - f[c]["rgb"][t].astype(int)))))
                    iio.imwrite(out / f"rgb_{t}.png", np.concatenate([obs[c + "_image"][::-1] for c in CAMERAS], 1))
                    if t == 0:
                        packet = camera_packet(env, obs)
                        world = packet["world_from_base"]
                        points = batch["points"][0].numpy()
                        selected = points[(points[:, 3] - np.maximum(points[:, 4], points[:, 5])) > .15]
                        xyz = selected[:, :3] @ world[:3, :3].T + world[:3, 3]
                        report["sensor_red_surface"] = dict(point_count=len(selected),
                            world_surface_mean_m=xyz.mean(0).tolist(),
                            evaluator_cube_center_m=env.object_positions()[0].tolist(),
                            note="Surface centroid differs from cube center; diagnostic truth never enters student inputs")
                        report["world_from_base"] = world.tolist()
                action = f["action"][t]
                recovered = ((action - norm["action_mean"]) / norm["action_std"]) * norm["action_std"] + norm["action_mean"]
                maximum["action_normalization_roundtrip"] = max(maximum["action_normalization_roundtrip"], float(np.max(np.abs(action - recovered))))
                obs, _, _, _ = env.step(action)
        assert maximum["state"] == maximum["rgb"] == maximum["points"] == 0
        assert maximum["action_normalization_roundtrip"] < 1e-6
        report["recorded_live_max_errors"] = maximum
        obs = env.reset_scene(110010)
        initial = obs["robot0_eef_pos"].copy()
        for _ in range(10):
            obs, _, _, _ = env.step(np.array([.5, 0, 0, 0, 0, 0, -1.]))
        displacement = obs["robot0_eef_pos"] - initial
        assert displacement[0] > .01
        for _ in range(20):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.]))
        opened = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        for _ in range(20):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.]))
        closed = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        assert opened > .06 and closed < .015
        report["physical_controller"] = dict(positive_x_displacement_m=displacement.tolist(),
                                              negative_command_width_m=opened,
                                              positive_command_width_m=closed)
        gates = []
        for model_name in ("direct_bc", "v1_diffusion"):
            torch.manual_seed(41)
            cfg["model"] = model_name
            model = build(cfg).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            batch = {k: v.cuda() for k, v in data.batch(np.random.default_rng(41), 32, norm).items()}
            loss, _ = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Resume the full state, then compare the actual next stochastic update.
            save(out / f"{model_name}_gate.pt", model, optimizer, None, norm, cfg,
                 dict(update=1), {})
            loss, _ = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            load(out / f"{model_name}_gate.pt", model, optimizer)
            loss, _ = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert all(torch.equal(expected[k], v.cpu()) for k, v in model.state_dict().items())
            gates.append(dict(model=model_name, actual_batch=32, renderer_cameras=list(CAMERAS),
                              next_update_exact=True, resources=watch.sample(force=True)))
            del model, optimizer, batch, expected
            torch.cuda.empty_cache()
        report["cuda_gates"] = gates
    finally:
        env.close()
    write(ROOT / "results/v3/initial_checks.json", report)
    print(json.dumps(report), flush=True)
    return report
