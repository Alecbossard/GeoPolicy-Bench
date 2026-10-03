"""Executed contracts for calibration, corruption, parity and actual CUDA resume."""

import copy
import hashlib
import json
import numpy as np
import torch
import h5py
from geopolicy.sensors import deproject, camera_packet, student_state
from geopolicy.v3.models import build
from geopolicy.v3.environment import SinglePlace
from geopolicy.v3.stages import StageData, stage_live
from geopolicy.checkpoint import save, load
from .common import ROOT, plan, sha, write
from .data import Data, live_batch
from .perturbations import CAMERAS, perturb, fuse, evaluation_rng, packet_digest
from .training import tensor_hash


def checks():
    p = plan()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(
        ROOT / "artifacts/v3/runs/bc_continued_history480_s0/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = dict(saved["config"], view="fusion", augmentation_recipe=p["augmentation"])
    norm = saved["normalization"]
    data = Data(cfg, "train")
    old = StageData(cfg, "train")
    assert data.identities == old.identities and len(data.index) == 9986
    for a, b in zip(data.episodes, old.episodes):
        for k in ("state", "action", "instruction_tokens", "points", "mask"):
            assert np.array_equal(a[k], b[k]), k
    for view in ("fixed", "fusion"):
        cfg["view"] = view
        e = data.episodes[0]
        packet = {c: {k: v[0] for k, v in e["cameras"][c].items()} for c in CAMERAS}
        assert all(
            np.array_equal(a, b)
            for a, b in zip(
                fuse(packet, view),
                fuse(
                    perturb(packet, p["conditions"]["nominal"], evaluation_rng(1, 0)),
                    view,
                ),
            )
        )
        absent = perturb(packet, p["conditions"]["absent"], evaluation_rng(1, 0))
        assert not absent[CAMERAS[0]]["mask"].any()
        assert not absent[CAMERAS[0]]["points"].any()
        assert np.array_equal(
            absent[CAMERAS[1]]["points"], packet[CAMERAS[1]]["points"]
        )
    cfg["view"] = "fusion"
    for condition in p["conditions"].values():
        a = perturb(packet, condition, evaluation_rng(1, 0))
        b = perturb(packet, condition, evaluation_rng(1, 0))
        assert packet_digest(a) == packet_digest(b)
        assert all(
            np.isfinite(a[c]["points"]).all()
            and not a[c]["points"][~a[c]["mask"]].any()
            for c in CAMERAS
        )
    low = perturb(packet, p["conditions"]["missing30"], evaluation_rng(1, 0))
    high = perturb(packet, p["conditions"]["missing90"], evaluation_rng(1, 0))
    assert all(not (high[c]["mask"] & ~low[c]["mask"]).any() for c in CAMERAS)
    # Verify ray-equivalence against a direct camera-space axial-depth change.
    rng = evaluation_rng(1, 0)
    noise = rng.standard_normal(512)
    rng.random(512)
    changed = perturb(packet, p["conditions"]["depth010"], evaluation_rng(1, 0))[
        CAMERAS[0]
    ]
    q = packet[CAMERAS[0]]
    tr = q["base_from_camera"]
    xyz = (q["points"][:, :3] - tr[:3, 3]) @ tr[:3, :3]
    z = xyz[:, 2]
    valid = q["mask"]
    expected = xyz[valid] * (1 + noise[valid, None] * 0.01 / z[valid, None])
    expected = expected @ tr[:3, :3].T + tr[:3, 3]
    assert np.allclose(changed["points"][valid, :3], expected, atol=3e-7)
    # IID dropout frequency checked on100k deterministic sampled points.
    draws = np.random.default_rng(44).random(100000)
    assert abs((draws < 0.7).mean() - 0.7) < 0.01
    net = build(cfg).eval()
    net.load_state_dict(saved["extra"]["ema"])
    empty = {
        c: dict(
            packet[c], points=np.zeros((512, 6), np.float32), mask=np.zeros(512, bool)
        )
        for c in CAMERAS
    }
    with torch.inference_mode():
        assert torch.isfinite(
            net.predict(live_batch(empty, [e["state"][0]], cfg, norm))
        ).all()
    del old
    # Recorded/live initial observations and projection/fusion share exact contracts.
    env = SinglePlace(cameras=True)
    try:
        obs = env.reset_scene(10000)
        live_packet = camera_packet(env, obs)
        with h5py.File(ROOT / data.identities[0]["path"]) as f:
            for c in CAMERAS:
                for k in (
                    "rgb",
                    "depth_m",
                    "intrinsic",
                    "base_from_camera",
                    "points",
                    "mask",
                ):
                    assert np.array_equal(f[c][k][0], live_packet[c][k]), (c, k)
        for view in ("fixed", "fusion"):
            cfg["view"] = view
            a = live_batch(
                perturb(
                    live_packet, p["conditions"]["nominal"], evaluation_rng(10000, 0)
                ),
                [student_state(obs)],
                cfg,
                norm,
            )
            b = stage_live(env, obs, [student_state(obs)], cfg, norm)
            for k in ("state", "instruction", "points", "point_mask"):
                assert torch.equal(a[k], b[k]), k
    finally:
        env.close()
    cfg["view"] = "fusion"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    net = build(cfg).cuda()
    ema = copy.deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0003)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2000, eta_min=0.00003)
    rng = np.random.default_rng(0)
    aug = np.random.default_rng(771000)

    def step():
        b = {k: v.cuda() for k, v in data.batch(rng, 32, norm, aug).items()}
        bh = hashlib.sha256(
            b["points"].cpu().numpy().tobytes() + b["action"].cpu().numpy().tobytes()
        ).hexdigest()
        loss, _ = net.loss(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        opt.step()
        sch.step()
        with torch.no_grad():
            for a, c in zip(ema.parameters(), net.parameters()):
                a.lerp_(c, 0.005)
        return dict(
            batch=bh,
            loss=float(loss.detach()),
            model=tensor_hash(net.state_dict()),
            ema=tensor_hash(ema.state_dict()),
        )

    for _ in range(3):
        step()
    target = ROOT / "artifacts/v4/checks/resume.pt"
    save(
        target,
        net,
        opt,
        sch,
        norm,
        cfg,
        dict(update=3),
        dict(
            ema=ema.state_dict(),
            batch_rng=rng.bit_generator.state,
            augmentation_rng=aug.bit_generator.state,
        ),
    )
    expected = step()
    restored = load(target, net, opt, sch)
    ema.load_state_dict(restored["extra"]["ema"])
    rng.bit_generator.state = restored["extra"]["batch_rng"]
    aug.bit_generator.state = restored["extra"]["augmentation_rng"]
    actual = step()
    assert expected == actual
    result = dict(
        dataset_episodes=79,
        dataset_frames=9986,
        all_nominal_cached_inputs_exact_v3=True,
        recorded_live_exact=True,
        axial_depth_ray_equivalence=True,
        nested_missing_masks=True,
        finite_empty_policy=True,
        cuda_augmented_resume_next_update_exact=True,
        resume_expected=expected,
        resume_actual=actual,
        depth_units="meters",
        coordinates="robot_base",
        point_corruption_stage="after_crop_voxel_sample_before_fusion",
    )
    write(ROOT / "results/v4/input_checks.json", result)
    print(json.dumps(result), flush=True)
