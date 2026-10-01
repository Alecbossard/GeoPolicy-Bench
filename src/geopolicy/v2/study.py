"""Serial, restartable V2 study; validation selection precedes a locked final test."""

import json
import subprocess
import sys
import time
from pathlib import Path
from geopolicy.io import save_json
from .config import file_hash, json_hash, source_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def status():
    """Read progress without importing Torch or starting a worker."""
    state_path = Path("artifacts/v2/study_stages.json")
    stages = read(state_path) if state_path.exists() else {}
    log = Path("artifacts/v2/study_pipeline.log")
    lines = log.read_text(encoding="utf8").splitlines() if log.exists() else []
    last = next(
        (s.removeprefix("V2 study ") for s in reversed(lines) if s.startswith("V2 study ")), None
    )
    out = dict(
        completed_main_trainings=sum(
            k.startswith("train/") and v["complete"] for k, v in stages.items()
        ),
        planned_main_trainings=21,
        latest_stage=last,
        failed_stages=[k for k, v in stages.items() if not v["complete"]],
        final_protocol_frozen=Path("configs/v2/final_protocol.json").exists(),
    )
    if last:
        worker = Path("artifacts/v2/study_logs") / (last.replace("/", "_") + ".log")
        if worker.exists():
            records = [
                s
                for s in worker.read_text(encoding="utf8").splitlines()
                if s.startswith('{"update"')
            ]
            if records:
                out["latest_update"] = json.loads(records[-1])["update"]
    for label, parent in [
        ("main_test", "test"),
        ("before_after", "before_after"),
        ("counterfactual", "counterfactual"),
    ]:
        out[label + "_completed_rollouts"] = sum(
            len(read(p)) for p in Path(f"artifacts/v2/{parent}").glob("*/rollouts.json")
        )
    return out


def stage(key, arguments, state, state_path, timeout=3600):
    if state.get(key, {}).get("complete"):
        return
    log = Path("artifacts/v2/study_logs") / (key.replace("/", "_") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "geopolicy.v2"] + arguments
    print("V2 study", key, flush=True)
    start = time.monotonic()
    with log.open("a", encoding="utf8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=timeout)
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
    state[key] = dict(
        complete=code == 0,
        returncode=code,
        wall_seconds=time.monotonic() - start,
        log=str(log),
        command=command,
    )
    save_json(state_path, state)
    with Path("PROGRESS.md").open("a", encoding="utf8") as stream:
        stream.write(f"\nV2 study {key}: {'complete' if code == 0 else 'FAILED'}; `{log}`.\n")
    if code:
        raise RuntimeError(f"Study stage failed: {key}; completed work preserved in {state_path}")


def selection(recipe):
    from .presets import PILOTS

    scores = {}
    for name in PILOTS:
        rows = read(f"artifacts/v2/pilot_evaluations/{name}/rollouts.json")
        count = recipe["evaluation"]["validation_episodes"]
        first = recipe["evaluation"]["validation_first_seed"]
        assert len(rows) == count and {r["scene_seed"] for r in rows} == set(
            range(first, first + count)
        )
        scores[name] = {
            "stable_successes": sum(r["stable_success"] for r in rows),
            "wrong_object_lifts": sum(r["wrong_object_lifted"] for r in rows),
            "collision_episodes": sum(bool(r.get("collision", False)) for r in rows),
            "transport_reached": sum(r["transport_reached"] for r in rows),
            "episodes": len(rows),
        }

    def rank(name):
        s = scores[name]
        return (
            -s["stable_successes"],
            s["wrong_object_lifts"],
            s["collision_episodes"],
            -s["transport_reached"],
            list(PILOTS).index(name),
        )

    chosen = {
        family: min([n for n in PILOTS if n.startswith(family)], key=rank)
        for family in ["diffusion", "act"]
    }
    report = dict(
        selected=chosen,
        pilot_scores=scores,
        rule=read("configs/v2/selection_rule.json"),
        reserved_test_used=False,
    )
    save_json("results/v2/selection.json", report)
    return report


def make_plan(recipe, chosen):
    from .presets import run_config

    jobs = []
    for view in ["fixed", "wrist", "fusion"]:
        for prior in [True, False]:
            name = f"{view}_{'prior' if prior else 'no_prior'}"
            for seed in recipe["training"]["seeds"]:
                cfg = run_config(
                    recipe,
                    chosen["diffusion"],
                    seed,
                    pilot=False,
                    overrides=dict(view=view, color_prior="chroma40" if prior else False),
                )
                cfg["name"] = name
                path = f"configs/v2/runs/{name}_s{seed}.json"
                save_json(path, cfg)
                jobs.append(
                    dict(
                        name=name,
                        seed=seed,
                        run_config=path,
                        checkpoint=f"artifacts/v2/main_runs/{name}_s{seed}/best.pt",
                    )
                )
    for seed in recipe["training"]["seeds"]:
        cfg = run_config(recipe, chosen["act"], seed, pilot=False)
        cfg["name"] = "act_selected"
        path = f"configs/v2/runs/act_selected_s{seed}.json"
        save_json(path, cfg)
        jobs.append(
            dict(
                name="act_selected",
                seed=seed,
                run_config=path,
                checkpoint=f"artifacts/v2/main_runs/act_selected_s{seed}/best.pt",
            )
        )
    save_json("configs/v2/main_plan.json", dict(recipe_sha256=json_hash(recipe), jobs=jobs))
    return jobs


def freeze_protocol(recipe, jobs, selected):
    from datetime import datetime, timezone

    target = Path("configs/v2/final_protocol.json")
    frozen_at = (
        read(target)["frozen_at_utc"] if target.exists() else datetime.now(timezone.utc).isoformat()
    )
    checkpoints = [j["checkpoint"] for j in jobs] + [
        f"artifacts/main_runs/{mode}_s{seed}/best.pt"
        for mode in ["fusion", "act"]
        for seed in recipe["training"]["seeds"]
    ]
    sources = [
        "src/geopolicy/environment.py",
        "src/geopolicy/sensors.py",
        "src/geopolicy/evaluation.py",
        "src/geopolicy/policies.py",
    ]
    sources += [
        f"src/geopolicy/v2/{name}.py"
        for name in ["evaluation", "metrics", "data", "models", "config", "resources"]
    ]
    protocol = dict(
        version=2,
        recipe=recipe,
        recipe_sha256=json_hash(recipe),
        selected=selected,
        jobs=jobs,
        registered_checkpoint_hashes=[file_hash(p) for p in checkpoints],
        checkpoints={p: file_hash(p) for p in checkpoints},
        evaluation_source_hashes={p: source_hash(p) for p in sources},
        evaluation_source_hash_algorithm="SHA256 of bytes with CRLF normalized to LF",
        frozen_at_utc=frozen_at,
        dataset_manifest_sha256=file_hash(recipe["dataset_manifest"]),
        normalization_sha256=file_hash(recipe["data_extension"]["normalization_path"]),
        primary_comparisons=[
            "fusion_prior vs fixed_prior, nominal",
            "fusion_prior vs fixed_prior, fixed_camera_missing",
            "fusion_prior vs wrist_prior, fixed_camera_missing",
            "fusion_prior vs fusion_no_prior, nominal",
        ],
        statistics="Paired scene and training-seed crossed bootstrap; 10000 draws, RNG 81, percentile 95% intervals; primary and exploratory comparisons are descriptive, without multiplicity correction.",
        confirmation=f"{recipe['evaluation']['confirmation_first_seed']}-{recipe['evaluation']['confirmation_first_seed']+recipe['evaluation']['confirmation_episodes']-1} reported without retuning",
        before_after="Preserved V1 fusion and ACT re-evaluated on the same new nominal scenes, horizon and stable metric; recipe changes are bundled, not attributed individually.",
        counterfactual="10 physically identical scenes x four instructions x three seeds for fusion prior on/off and selected ACT; all-four completion is the strict score.",
        frozen_before_first_reserved_rollout=True,
    )
    if target.exists():
        assert read(target) == protocol, "Cannot change an already frozen protocol"
    else:
        save_json(target, protocol)
    return protocol


def run_study(recipe):
    state_path = Path("artifacts/v2/study_stages.json")
    state = read(state_path) if state_path.exists() else {}
    from .presets import PILOTS

    while True:
        pilots = (
            read("artifacts/v2/pilot_stages.json")
            if Path("artifacts/v2/pilot_stages.json").exists()
            else {}
        )
        if all(pilots.get(f"{name}/evaluate", {}).get("complete") for name in PILOTS):
            break
        if any(not v["complete"] for v in pilots.values()):
            raise RuntimeError("A pilot failed; repair/resume pilots before the main study")
        time.sleep(10)
    stage("diagnostics/input_sensitivity", ["inspect-act"], state, state_path)
    control = "artifacts/v2/pilots/diffusion_original_data"
    args = [
        "--config",
        "configs/v2/original_data_recipe.json",
        "train",
        "--preset",
        "diffusion_suffix_only",
        "--out",
        control,
    ]
    if Path(control + "/latest.pt").exists():
        args += ["--resume", control + "/latest.pt"]
    stage("control/original_data_train", args, state, state_path)
    stage(
        "control/original_data_evaluate",
        [
            "evaluate",
            "--checkpoint",
            control + "/best.pt",
            "--out",
            "artifacts/v2/pilot_evaluations/diffusion_original_data",
        ],
        state,
        state_path,
    )
    selected = selection(recipe)
    for family, name in selected["selected"].items():
        stage(
            f"confirmation/{family}",
            [
                "evaluate",
                "--checkpoint",
                f"artifacts/v2/pilots/{name}/best.pt",
                "--out",
                f"artifacts/v2/confirmation/{family}",
                "--first-seed",
                str(recipe["evaluation"]["confirmation_first_seed"]),
                "--episodes",
                str(recipe["evaluation"]["confirmation_episodes"]),
            ],
            state,
            state_path,
        )
    jobs = make_plan(recipe, selected["selected"])
    for job in jobs:
        name = f"{job['name']}_s{job['seed']}"
        folder = str(Path(job["checkpoint"]).parent)
        args = ["train", "--run-config", job["run_config"], "--out", folder]
        if Path(folder + "/latest.pt").exists():
            args += ["--resume", folder + "/latest.pt"]
        stage(f"train/{name}", args, state, state_path)
    # Run unit tests and a real checkpoint continuation before opening the final test.
    stage("verification/pretest", ["verify"], state, state_path)
    freeze_protocol(recipe, jobs, selected)
    e = recipe["evaluation"]
    for job in jobs:
        name = f"{job['name']}_s{job['seed']}"
        stage(
            f"test/{name}",
            [
                "evaluate",
                "--checkpoint",
                job["checkpoint"],
                "--out",
                f"artifacts/v2/test/{name}",
                "--first-seed",
                str(e["reserved_test_first_seed"]),
                "--episodes",
                str(e["test_episodes"]),
                "--conditions",
            ]
            + e["conditions"]
            + ["--reserved-test"],
            state,
            state_path,
        )
    for mode in ["fusion", "act"]:
        for seed in recipe["training"]["seeds"]:
            stage(
                f"before_after/{mode}_s{seed}",
                [
                    "evaluate",
                    "--checkpoint",
                    f"artifacts/main_runs/{mode}_s{seed}/best.pt",
                    "--out",
                    f"artifacts/v2/before_after/{mode}_s{seed}",
                    "--first-seed",
                    str(e["reserved_test_first_seed"]),
                    "--episodes",
                    str(e["test_episodes"]),
                    "--reserved-test",
                ],
                state,
                state_path,
            )
    for name in ["fusion_prior", "fusion_no_prior", "act_selected"]:
        for seed in recipe["training"]["seeds"]:
            for obj in [0, 1]:
                for goal in [0, 1]:
                    cell = f"{name}_s{seed}_o{obj}_g{goal}"
                    stage(
                        f"counterfactual/{cell}",
                        [
                            "evaluate",
                            "--checkpoint",
                            f"artifacts/v2/main_runs/{name}_s{seed}/best.pt",
                            "--out",
                            f"artifacts/v2/counterfactual/{cell}",
                            "--first-seed",
                            str(e["counterfactual_first_seed"]),
                            "--episodes",
                            str(e["counterfactual_scenes"]),
                            "--object-id",
                            str(obj),
                            "--goal-id",
                            str(goal),
                            "--reserved-test",
                        ],
                        state,
                        state_path,
                    )
    stage("delivery/report", ["report"], state, state_path)
    print(
        "Study completed; review the report and delivery verification before declaring V2 complete",
        flush=True,
    )
