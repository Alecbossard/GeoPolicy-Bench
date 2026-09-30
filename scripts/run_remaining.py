"""Serial remaining experiments; no implicit publication, resumable stages."""

import json
import subprocess
import sys
import time
from pathlib import Path

main = Path("artifacts/main_runs")
state_path = Path("artifacts/remaining_stages.json")
state = json.loads(state_path.read_text()) if state_path.exists() else {}
python = sys.executable
vla = str(Path(".venv-vla/Scripts/python.exe"))
repro = str(Path(".venv-repro/Scripts/python.exe"))


def run(name, cmd):
    if state.get(name, {}).get("complete"):
        print("Stage complete", name, flush=True)
        return
    print("Stage", name, "command", cmd, flush=True)
    start = time.time()
    log = Path("artifacts") / f"remaining_{name}.log"
    with log.open("a") as stream:
        code = subprocess.call(cmd, stdout=stream, stderr=subprocess.STDOUT)
    state[name] = {
        "complete": code == 0,
        "returncode": code,
        "wall_seconds": time.time() - start,
        "command": cmd,
        "log": str(log),
    }
    state_path.write_text(json.dumps(state, indent=2))
    with Path("PROGRESS.md").open("a") as stream:
        stream.write(
            f"\nRemaining-stage {name}: {'complete' if code==0 else 'FAILED'}, wall{time.time()-start:.1f}s, log{log}.\n"
        )
    if code:
        raise RuntimeError(f"Stage failed: {name}; see {log}")


expected = [
    main / f"{m}_s{s}" / "manifest.json" for m in ["fusion", "mono", "act"] for s in [0, 1, 2]
]
print("Waiting for all nine principal trainings before GPU-dependent remaining stages", flush=True)
while not all(p.exists() for p in expected):
    time.sleep(15)
# Child writes manifest immediately before exiting; allow the CUDA context to close.
time.sleep(15)
run("clean_smoke", [repro, "scripts/smoke_lift.py", "--out", "artifacts/clean_smoke_lift"])
run(
    "clean_replay",
    [
        repro,
        "-m",
        "geopolicy.cli",
        "evaluate",
        "--checkpoint",
        str(main / "fusion_s0/best.pt"),
        "--out",
        "artifacts/clean_checkpoint_replay",
        "--episodes",
        "1",
        "--first-seed",
        "100111",
        "--conditions",
        "nominal",
        "--device",
        "cpu",
        "--execute-steps",
        "2",
        "--videos",
        "1",
    ],
)
for mode in ["fusion", "mono", "act"]:
    run(
        f"profile_{mode}",
        [
            python,
            "scripts/profile_policy.py",
            "--checkpoint",
            str(main / f"{mode}_s0/best.pt"),
            "--out",
            f"results/profile_{mode}.json",
        ],
    )
run("secondary_training", [python, "scripts/run_secondary_training.py"])
run("main_evaluation", [python, "scripts/run_evaluation_matrix.py"])
run(
    "counterfactual_evaluation",
    [python, "scripts/run_evaluation_matrix.py", "--stage", "counterfactual"],
)
for name, checkpoint, conditions, extra in [
    ("data50", "artifacts/secondary_runs/fusion_50d_s0/best.pt", ["nominal"], []),
    (
        "view_dropout",
        "artifacts/secondary_runs/fusion_dropout_s0/best.pt",
        ["nominal", "fixed_camera_missing"],
        [],
    ),
    ("no_voxel", "artifacts/main_runs/fusion_s0/best.pt", ["nominal"], ["--voxel", "0"]),
    (
        "onnx",
        "artifacts/main_runs/fusion_s0/best.pt",
        ["nominal"],
        ["--onnx-path", "artifacts/onnx_export/denoiser.onnx"],
    ),
]:
    run(
        f"evaluate_{name}",
        [
            python,
            "-m",
            "geopolicy.cli",
            "evaluate",
            "--checkpoint",
            checkpoint,
            "--out",
            f"artifacts/secondary_evaluations/{name}",
            "--episodes",
            "20",
            "--first-seed",
            "200000",
            "--device",
            "cpu",
            "--execute-steps",
            "2",
            "--conditions",
        ]
        + conditions
        + extra
        + ["--videos", "1"],
    )
run(
    "smolvla_nominal",
    [
        vla,
        "scripts/smolvla_evaluate.py",
        "--out",
        "artifacts/smolvla_test",
        "--episodes",
        "20",
        "--first-seed",
        "200000",
        "--conditions",
        "nominal",
        "--videos",
        "1",
    ],
)
vla_rows = json.loads(Path("artifacts/smolvla_test/rollouts.json").read_text())
if any(r["success"] for r in vla_rows):
    run(
        "smolvla_ood",
        [
            vla,
            "scripts/smolvla_evaluate.py",
            "--out",
            "artifacts/smolvla_test_ood",
            "--episodes",
            "20",
            "--first-seed",
            "200000",
            "--conditions",
            "occlusion",
            "depth_degraded",
            "fixed_camera_missing",
            "extrinsic_error",
            "--videos",
            "0",
        ],
    )
else:
    state["smolvla_ood"] = {
        "deferred": True,
        "reason": "0/20 nominal success; conditional secondary robustness budget deferred per frozen protocol",
    }
    state_path.write_text(json.dumps(state, indent=2))
run("summarize_primary", [python, "scripts/summarize_results.py"])
print(
    "All implemented remaining experiment stages complete; final secondary aggregation/documentation/visual QA still required",
    flush=True,
)
