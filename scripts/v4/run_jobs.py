"""Serial local worker, log/progress saved per cell, never two heavy processes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf8")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    args = p.parse_args()
    jobs_path = ROOT / args.jobs
    assert jobs_path.relative_to(ROOT).as_posix().startswith("configs/v4/")
    jobs = json.loads(jobs_path.read_text(encoding="utf8"))
    phase = jobs_path.stem
    budget_path = ROOT / "artifacts/v4/budget.json"
    if budget_path.exists():
        budget = json.loads(budget_path.read_text(encoding="utf8"))
    else:
        old_phases = list((ROOT / "artifacts/v4/jobs").glob("*.json"))
        starts = [
            json.loads(f.read_text(encoding="utf8"))["started_time"] for f in old_phases
        ]
        budget = dict(
            started_time=min(starts) if starts else time.time(),
            maximum_wall_seconds=12 * 3600,
            maximum_training_updates=32000,
            diagnostic_updates=5,
        )
        write(budget_path, budget)
    target = ROOT / "artifacts/v4/jobs" / f"{phase}.json"
    status = (
        json.loads(target.read_text(encoding="utf8"))
        if target.exists()
        else dict(
            jobs_sha256=__import__("hashlib")
            .sha256(jobs_path.read_bytes())
            .hexdigest(),
            started_time=time.time(),
            completed=[],
        )
    )
    assert (
        status["jobs_sha256"]
        == __import__("hashlib").sha256(jobs_path.read_bytes()).hexdigest()
    )
    for index, job in enumerate(jobs):
        if index in status["completed"]:
            continue
        if time.time() - budget["started_time"] > budget["maximum_wall_seconds"]:
            raise RuntimeError("Global V4 wall budget exhausted, progression saved")
        cli = job[:]
        if cli[0] == "train":
            view = cli[cli.index("--view") + 1]
            seed = cli[cli.index("--seed") + 1]
            group = view + ("_aug" if "--augmented" in cli else "_clean")
            scope = cli[cli.index("--scope") + 1] if "--scope" in cli else "pilot"
            directory = "main_runs" if scope == "main" else "runs"
            latest = ROOT / f"artifacts/v4/{directory}/{group}_s{seed}/latest.pt"
            total = budget["diagnostic_updates"]
            completed_here = 0
            manifests = list(
                (ROOT / "artifacts/v4/runs").glob("*/manifest.json")
            ) + list((ROOT / "artifacts/v4/main_runs").glob("*/manifest.json"))
            for manifest_path in manifests:
                value = json.loads(manifest_path.read_text(encoding="utf8"))[
                    "completed_update"
                ]
                total += value
                if manifest_path.parent == latest.parent:
                    completed_here = value
            wanted = (
                int(cli[cli.index("--stop-at") + 1]) if "--stop-at" in cli else 2000
            )
            assert (
                total + max(0, wanted - completed_here)
                <= budget["maximum_training_updates"]
            ), "Global training update budget exceeded"
            if latest.exists() and "--resume" not in cli:
                cli.append("--resume")
        log = ROOT / f"artifacts/v4/jobs/logs/{phase}_{index:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        status.update(
            current=index,
            current_job=cli,
            current_log=str(log.relative_to(ROOT)),
            current_started=time.time(),
        )
        write(target, status)
        print(
            json.dumps(dict(phase=phase, index=index, total=len(jobs), job=cli)),
            flush=True,
        )
        with log.open("a", encoding="utf8") as f:
            result = subprocess.run(
                [str(ROOT / ".venv/Scripts/python.exe"), "-m", "geopolicy.v4", *cli],
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUTF8="1"),
            )
        if result.returncode:
            status.update(failed=index, returncode=result.returncode)
            write(target, status)
            print(log.read_text(encoding="utf8", errors="replace")[-4000:], flush=True)
            sys.exit(result.returncode)
        status["completed"].append(index)
        status["last_completed_time"] = time.time()
        write(target, status)
    print(f"Completed {phase}: {len(jobs)} serial jobs", flush=True)


if __name__ == "__main__":
    main()
