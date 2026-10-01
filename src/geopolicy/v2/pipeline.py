"""One serial, resumable supervisor. Scientific settings come from the recipe."""

import json
import subprocess
import sys
import time
from pathlib import Path
from geopolicy.io import save_json
from .config import read_recipe


def run_pilots(recipe):
    state_path = Path("artifacts/v2/pilot_stages.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    print("Waiting for complete post-release dataset before any optimizer job", flush=True)
    while not Path(recipe["dataset_manifest"]).exists():
        time.sleep(10)
    time.sleep(3)
    from .presets import PILOTS

    for name in PILOTS:
        for stage in ["gate", "train", "evaluate"]:
            key = f"{name}/{stage}"
            if state.get(key, {}).get("complete"):
                continue
            base = [sys.executable, "-m", "geopolicy.v2"]
            if stage == "gate":
                cmd = base + ["gate", "--preset", name]
            elif stage == "train":
                cmd = base + ["train", "--preset", name, "--out", f"artifacts/v2/pilots/{name}"]
                latest = Path(f"artifacts/v2/pilots/{name}/latest.pt")
                if latest.exists():
                    cmd += ["--resume", str(latest)]
            else:
                cmd = base + [
                    "evaluate",
                    "--checkpoint",
                    f"artifacts/v2/pilots/{name}/best.pt",
                    "--out",
                    f"artifacts/v2/pilot_evaluations/{name}",
                    "--first-seed",
                    str(recipe["evaluation"]["validation_first_seed"]),
                    "--episodes",
                    str(recipe["evaluation"]["validation_episodes"]),
                ]
            log = Path("artifacts/v2") / f"pilot_{name}_{stage}.log"
            print("Pilot", key, flush=True)
            start = time.monotonic()
            with log.open("a", encoding="utf8") as stream:
                child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    code = child.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    import psutil

                    targets = psutil.Process(child.pid).children(recursive=True) + [
                        psutil.Process(child.pid)
                    ]
                    for process in reversed(targets):
                        try:
                            process.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    code = -1
            state[key] = {
                "complete": code == 0,
                "returncode": code,
                "wall_seconds": time.monotonic() - start,
                "log": str(log),
                "command": cmd,
            }
            save_json(state_path, state)
            with Path("PROGRESS.md").open("a", encoding="utf8") as stream:
                stream.write(
                    f"\nV2 pilot {key}: {'complete' if code==0 else 'FAILED'}; log `{log}`.\n"
                )
            if code:
                raise RuntimeError(f"Pilot stage failed: {key}; see {log}")
    print(
        "All six targeted pilots complete; selection/confirmation and frozen main experiments remain",
        flush=True,
    )
