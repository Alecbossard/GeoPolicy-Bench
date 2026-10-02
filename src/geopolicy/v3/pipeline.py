"""Serial local supervisor. Child checkpoints/rows support explicit safe resumption."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from .common import ROOT, write


def baseline():
    stages = [
        ("diffusion_four_tuning", ["evaluate", "--name", "diffusion_four_demos_tuning_s0",
          "--checkpoint", "artifacts/v3/runs/diffusion_overfit_s0/best.pt", "--episodes", "20"]),
        ("collect_train80", ["collect", "--first", "10000", "--episodes", "80"]),
        ("collect_validation10", ["collect", "--first", "110000", "--episodes", "10"]),
    ]
    for model, short in [("direct_bc", "bc"), ("v1_diffusion", "diffusion")]:
        name = f"{short}_original80_s0"
        stages.extend([
            (name + "_train", ["train", "--name", name, "--model", model, "--limit", "80"]),
            (name + "_tuning", ["evaluate", "--name", name + "_tuning",
             "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]),
        ])
    supervise("baseline", stages)


def supervise(group, stages):
    out = ROOT / "artifacts/v3/pipeline" / group
    out.mkdir(parents=True, exist_ok=True)
    status = out / "stages.json"
    saved = json.loads(status.read_text()) if status.exists() else {}
    for name, args in stages:
        if saved.get(name, {}).get("status") == "complete":
            continue
        # Continue an incomplete training with its immutable config and RNG.
        if args[0] == "train":
            run_name = args[args.index("--name") + 1]
            run = ROOT / "artifacts/v3/runs" / run_name
            if (run / "latest.pt").exists() and "--resume" not in args:
                args = args + ["--resume"]
        saved[name] = dict(status="running", args=args, started_utc=datetime.now(timezone.utc).isoformat())
        write(status, saved)
        print(f"Starting {group}/{name}", flush=True)
        with (out / f"{name}.log").open("a", encoding="utf8") as stream:
            process = subprocess.Popen([sys.executable, "-m", "geopolicy.v3", *args], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT)
            saved[name]["pid"] = process.pid
            write(status, saved)
            code = process.wait()
        saved[name].update(status="complete" if code == 0 else "failed", returncode=code,
                           finished_utc=datetime.now(timezone.utc).isoformat())
        write(status, saved)
        print(f"Finished {group}/{name}: {code}", flush=True)
        if code:
            raise RuntimeError(f"Stage {name} failed; checkpoints and log retained")
