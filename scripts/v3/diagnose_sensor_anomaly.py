"""Post-test quality check; preserves every original rollout and frozen recipe."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from geopolicy.v3.resources import preflight

    preflight("sensor_identity_quality_check")
    import numpy as np
    import torch
    from geopolicy.environment import CAMERAS
    from geopolicy.sensors import student_state
    from geopolicy.v3.common import write
    from geopolicy.v3.final_protocol import registered, json_hash
    from geopolicy.v3.environment import SinglePlace
    from geopolicy.v3.metrics import Tracker
    from geopolicy.v3.models import build
    from geopolicy.v3.data import live
    from geopolicy.v2.resources import ResourceWatch

    torch.set_num_threads(2)
    out = ROOT / "artifacts/v3/diagnostics/sensor_anomaly"
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    snapshots, checks = {}, {}
    protocol, _ = registered("single_wrist_prior_s0")
    original_result_hashes = {
        r["key"]: hashlib.sha256(
            (ROOT / f"results/v3/test/final_{r['key']}.json").read_bytes()
        ).hexdigest()
        for r in protocol["registry"]
    }
    # Reproduce the complete preceding sequence, not only an isolated reset.
    for key in ("single_wrist_prior_s0", "single_fixed_prior_s0"):
        protocol, record = registered(key)
        saved = torch.load(
            ROOT / record["checkpoint"], map_location="cpu", weights_only=False
        )
        cfg, norm = saved["config"], saved["normalization"]
        model = build(cfg).eval()
        model.load_state_dict(saved["extra"]["ema"])
        original = json.loads((ROOT / f"results/v3/test/final_{key}.json").read_text())[
            "rollouts"
        ]
        env = SinglePlace(cameras=True)
        env.horizon = plan["maximum_steps"]
        watch = ResourceWatch(out / key, plan["resource_limits"])
        rows = []
        try:
            for scene in range(400000, 400015):
                obs = env.reset_scene(scene)
                arrays = {"state": student_state(obs).copy()}
                for camera in CAMERAS:
                    for field in ("image", "depth"):
                        arrays[camera + "_" + field] = obs[camera + "_" + field].copy()
                digest = hashlib.sha256(arrays["state"].tobytes())
                for camera in CAMERAS:
                    digest.update(arrays[camera + "_image"].tobytes())
                    digest.update(arrays[camera + "_depth"].tobytes())
                if scene == 400014:
                    snapshots[key] = arrays
                    (out / key).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / key / "initial_400014.npz", **arrays)
                torch.manual_seed(scene + cfg["seed"] * 1000000)
                history, queue = [], []
                tracker = Tracker(plan["stability"])
                for step in range(env.horizon):
                    watch.sample()
                    history.append(student_state(obs).copy())
                    if not queue:
                        batch = live(env, obs, history, cfg, norm)
                        with torch.inference_mode():
                            prediction = model.predict(batch)[
                                0, : cfg["execute_steps"]
                            ].numpy()
                        queue = list(
                            np.clip(
                                prediction * norm["action_std"] + norm["action_mean"],
                                -1,
                                1,
                            )
                        )
                    action = queue.pop(0)
                    obs, _, done, info = env.step(action)
                    if tracker.update(env, obs, action, info) or done:
                        break
                old_trace = json.loads(
                    (
                        ROOT
                        / f"artifacts/v3/evaluations/final_{key}/traces/{scene}.json"
                    ).read_text()
                )
                row = dict(
                    scene_seed=scene,
                    initial_sensor_sha256=digest.hexdigest(),
                    original_initial_sensor_sha256=original[scene - 400000][
                        "initial_sensor_sha256"
                    ],
                    actions_physics_exact=json_hash(tracker.trace)
                    == json_hash(old_trace),
                    **tracker.summary(),
                )
                rows.append(row)
                write(out / key / "replay_checks.json", rows)
                if scene == 400014:
                    write(out / key / "trace_400014.json", tracker.trace)
                print(key, scene, row["actions_physics_exact"], flush=True)
        finally:
            env.close()
        checks[key] = rows
    differences = {}
    a, b = snapshots.values()
    for field in a:
        differences[field] = dict(
            equal=bool(np.array_equal(a[field], b[field])),
            differing_values=int(np.count_nonzero(a[field] != b[field])),
            max_abs_error=float(
                np.max(
                    np.abs(a[field].astype(np.float64) - b[field].astype(np.float64))
                )
            ),
        )
    reference = json.loads(
        (ROOT / "results/v3/test/final_single_fixed_prior_s0.json").read_text()
    )["rollouts"]
    mismatches = []
    for record in protocol["registry"]:
        rows = json.loads(
            (ROOT / f"results/v3/test/final_{record['key']}.json").read_text()
        )["rollouts"]
        for a, b in zip(rows, reference):
            if a["initial_sensor_sha256"] != b["initial_sensor_sha256"]:
                mismatches.append(
                    dict(
                        key=record["key"],
                        scene_seed=a["scene_seed"],
                        observed_sha256=a["initial_sensor_sha256"],
                        reference_sha256=b["initial_sensor_sha256"],
                    )
                )
    after_result_hashes = {
        r["key"]: hashlib.sha256(
            (ROOT / f"results/v3/test/final_{r['key']}.json").read_bytes()
        ).hexdigest()
        for r in protocol["registry"]
    }
    assert original_result_hashes == after_result_hashes
    result = dict(
        raw_mismatches=mismatches,
        original_rows_changed=False,
        original_result_hashes_before_after_exact=original_result_hashes,
        diagnostic_only_no_tuning=True,
        fresh_sequence_replays=checks,
        fresh_initial_modality_comparison=differences,
        limitation="Original per-modality initial arrays were not saved, only their combined digest. A transient original mismatch cannot be localized uniquely from that digest. Main raw scores retain every scene. Paired sensitivity contrasts involving a mismatch exclude that scene across all seeds, independently of success.",
    )
    write(ROOT / "results/v3/sensor_identity_quality_check.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "fresh_sequence_replays"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
