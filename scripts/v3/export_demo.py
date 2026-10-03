"""Export a small local inference checkpoint plus frozen source/config bundle."""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from geopolicy.v3.resources import preflight

    preflight("export_compact_demo")
    import torch
    from geopolicy.v3.common import sha, write
    from geopolicy.v3.final_protocol import registered, json_hash, tensor_hash

    key = "single_fusion_prior_s0"
    protocol, record = registered(key)
    row_path = ROOT / f"results/v3/test/final_{key}.json"
    rows = json.loads(row_path.read_text())["rollouts"]
    assert len(rows) == protocol["episodes"]
    row = rows[0]
    assert row["scene_seed"] == protocol["first"]
    trace = json.loads(
        (
            ROOT
            / f"artifacts/v3/evaluations/final_{key}/traces/{row['scene_seed']}.json"
        ).read_text()
    )
    saved = torch.load(
        ROOT / record["checkpoint"], map_location="cpu", weights_only=False
    )
    compact = dict(
        config=saved["config"],
        normalization=saved["normalization"],
        extra=dict(ema=saved["extra"]["ema"]),
    )
    assert tensor_hash(compact["extra"]["ema"]) == record["ema_tensor_sha256"]
    out = ROOT / "artifacts/v3/demo"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(compact, out / "checkpoint.pt")
    keys = (
        "approached",
        "grasped",
        "lifted",
        "transported",
        "release_seen",
        "physical_success",
        "strict_v2_success",
        "posture_only_failure",
        "failure_stage",
        "physical_dwell_s",
        "strict_dwell_s",
    )
    proof = dict(
        key=key,
        scene_seed=row["scene_seed"],
        source_checkpoint_sha256=record["checkpoint_sha256"],
        compact_sha256=sha(out / "checkpoint.pt"),
        compact_bytes=(out / "checkpoint.pt").stat().st_size,
        ema_tensor_exact=True,
        initial_sensor_sha256=row["initial_sensor_sha256"],
        frozen_trace_sha256=json_hash(trace),
        frozen_summary={k: row[k] for k in keys},
        selection="First prespecified test scene and training seed0, chosen before test; no cherry-picked scene",
    )
    write(out / "equivalence.json", proof)
    write(ROOT / "configs/v3/demo_equivalence.json", proof)
    bundle = ROOT / "artifacts/v3/portable_demo"
    assert bundle.resolve().is_relative_to((ROOT / "artifacts/v3").resolve())
    shutil.copytree(
        ROOT / "src/geopolicy",
        bundle / "src/geopolicy",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (bundle / "configs/v3").mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "runtime_limits.json", "final_protocol.json"):
        shutil.copy2(ROOT / "configs/v3" / name, bundle / "configs/v3" / name)
    for name in ("checkpoint.pt", "equivalence.json"):
        shutil.copy2(out / name, bundle / name)
    shutil.copy2(ROOT / "scripts/v3/replay_demo.py", bundle / "replay_demo.py")
    shutil.copy2(ROOT / "requirements-lock.txt", bundle / "requirements-lock.txt")
    write(bundle / "expected_trace.json", trace)
    write(bundle / "expected_rollout.json", row)
    (bundle / "README.md").write_text(
        "# GeoPolicy-Bench V3 local demo\n\n"
        "Run `path/to/pinned/python.exe replay_demo.py` from this extracted folder.\n"
        "The script explicitly imports this bundle's source and requires Python3.11.9, "
        "Torch2.7.1+cu128, NumPy1.26.4, MuJoCo3.3.7, robosuite1.5.2 and h5py3.14.0. "
        "The included requirements lock records the existing environment.\n\n"
        "Inputs: learned EMA checkpoint, robot state, simulated RGB-D points, fixed "
        "known instruction labels. No ground-truth object pose or teacher phase is "
        "given to the student. Output: replay/demo.mp4, trace.json and verification.json.\n\n"
        "Scene400000 and training seed0 were chosen before the test. The replay "
        "requires the complete action/physics trace to match the frozen test exactly. "
        "The expected trace is included for inspection. No training datasets, "
        "optimizer or historical checkpoints are needed.\n\n"
        "Scope: one cube / one tray / fixed instruction in simulation. "
        "Multi-object competence and real-robot transfer are not established. "
        "This is one example; the project's raw results contain all1500 final rollouts.\n",
        encoding="utf8",
    )
    write(
        ROOT / "results/v3/demo_export.json",
        dict(
            **proof,
            bundle="artifacts/v3/portable_demo",
            replay_script_sha256=sha(ROOT / "scripts/v3/replay_demo.py"),
        ),
    )
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
