"""Wait for experimental pipeline, then serial verification and report artifacts."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from geopolicy.io import save_json
from job_runtime import stop_owned_job

state_path = Path("artifacts/finalization_stages.json")
state = json.loads(state_path.read_text()) if state_path.exists() else {}
print("Waiting for all experimental stages, including VLA, before combined GPU gates", flush=True)
while True:
    try:
        experiments = json.loads(Path("artifacts/remaining_stages.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        experiments = {}
    if experiments.get("summarize_primary", {}).get("complete"):
        break
    if any(s.get("returncode", 0) != 0 for s in experiments.values() if isinstance(s, dict)):
        raise RuntimeError(
            "Experimental pipeline failed; preserve completed cells and repair before finalization"
        )
    time.sleep(15)
time.sleep(15)


def run(name, cmd, env=None):
    if state.get(name, {}).get("complete"):
        return
    print("Finalization", name, cmd, flush=True)
    start = time.time()
    log = Path(f"artifacts/finalize_{name}.log")
    with log.open("a") as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
        try:
            code = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            stop_owned_job(proc)
            code = -1
    state[name] = {
        "complete": code == 0,
        "returncode": code,
        "wall_seconds": time.time() - start,
        "log": str(log),
        "command": cmd,
    }
    save_json(state_path, state)
    with Path("PROGRESS.md").open("a", encoding="utf8") as stream:
        stream.write(f"\nFinalization{name}: {'complete' if code==0 else 'FAILED'},log{log}.\n")
    if code:
        raise RuntimeError(f"Finalization failed: {name}; see {log}")


py = sys.executable
vla = str(Path(".venv-vla/Scripts/python.exe"))
repro = str(Path(".venv-repro/Scripts/python.exe"))
run("combined_compact", [py, "scripts/combined_resource_gate.py"])
run("combined_vla", [vla, "scripts/combined_resource_gate.py", "--vla"])
gpu_env = dict(os.environ)
gpu_env["GEO_GPU_TESTS"] = "1"
run("gpu_resume", [py, "-m", "pytest", "tests/test_gpu_resume.py", "-q"], gpu_env)
run("full_training_resume", [py, "scripts/verify_full_training_resume.py"])
cpu_env = dict(os.environ)
cpu_env.pop("GEO_GPU_TESTS", None)
run("clean_tests", [repro, "-m", "pytest", "-q"], cpu_env)
run("secondary_summary", [py, "scripts/summarize_secondary.py"])
run("final_report", [py, "scripts/write_final_report.py"])
print(
    "Experiment and automated verification artifacts complete; human-agent visual QA/Git/readme audit remains",
    flush=True,
)
