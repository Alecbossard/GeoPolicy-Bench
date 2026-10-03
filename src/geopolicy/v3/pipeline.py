"""Serial local supervisor. Child checkpoints/rows support explicit safe resumption."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from .common import ROOT, write


def validation_counts(result, seed, first):
    """Only complete, unassisted held-out validation can open a competence gate."""
    rows = result["rollouts"]
    identity = result["identity"]
    assert 110100 <= first < 200000 and identity["first"] == first
    assert identity["episodes"] == 20 and len(rows) == 20
    assert not identity["reference"] and not identity["raw_weights"]
    assert identity["checkpoint_sha256"]
    assert [r["scene_seed"] for r in rows] == list(range(first, first + 20))
    assert all(r["training_seed"] == seed and not r["diagnostic_oracle"]
               and not r["overfit_diagnostic"]
               and r["checkpoint_sha256"] == identity["checkpoint_sha256"] for r in rows)
    return dict(physical=sum(r["physical_success"] for r in rows),
                strict=sum(r["strict_v2_success"] for r in rows))


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


def confirmation():
    stages = []
    for model, short in [("direct_bc", "bc"), ("v1_diffusion", "diffusion")]:
        rows = json.loads((ROOT / f"results/v3/validation/{short}_original80_s0_tuning.json").read_text())["rollouts"]
        if sum(r["physical_success"] for r in rows) < 12:
            print(f"{short}: seed 0 did not pass the competence gate; no repetition matrix", flush=True)
            continue
        for seed in (0, 1, 2):
            name = f"{short}_original80_s{seed}"
            if seed:
                stages.extend([
                    (name + "_train", ["train", "--name", name, "--model", model,
                     "--limit", "80", "--seed", str(seed)]),
                    (name + "_tuning", ["evaluate", "--name", name + "_tuning",
                     "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]),
                ])
            stages.append((name + "_confirmation", ["evaluate", "--name", name + "_confirmation",
                           "--first", "110200", "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]))
    supervise("confirmation", stages)


def competence(short="bc"):
    results = []
    for seed in (0, 1, 2):
        for split in ("tuning", "confirmation"):
            path = ROOT / f"results/v3/validation/{short}_original80_s{seed}_{split}.json"
            assert path.exists(), f"Gate pending: {path.name}"
            result = json.loads(path.read_text())
            count = validation_counts(result, seed, 110100 if split == "tuning" else 110200)["physical"]
            results.append(dict(model=short, seed=seed, split=split, successes=count, episodes=20))
            assert count >= 12, f"Competence gate failed: {short} seed {seed} {split}: {count}/20"
    write(ROOT / "results/v3/competence_gate.json", results)
    return results


def ablations():
    competence("bc")
    stages = []
    for variant, extra in [("continued", ["--continued"]),
                           ("state_history4", ["--history", "4"]),
                           ("binary", ["--binary"])]:
        for seed in (0, 1, 2):
            name = f"bc_{variant}80_s{seed}"
            stages.extend([
                (name + "_train", ["train", "--name", name, "--limit", "80", "--seed", str(seed), *extra]),
                (name + "_tuning", ["evaluate", "--name", name + "_tuning",
                 "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]),
            ])
    supervise("ablations", stages)


def progressive(task="two_objects_one_goal"):
    competence("bc")
    selection = json.loads((ROOT / "configs/v3/selection.json").read_text())
    if task == "full":
        path = ROOT / "configs/v3/two_objects_one_goal_selection.json"
        if path.exists():
            selection = json.loads(path.read_text())
    assert selection["reserved_test_used"] is False
    recipe = selection["recipe"]
    extra = ["--history", str(selection["history"]), "--model", selection.get("model", "direct_bc")]
    if selection["continued"]:
        extra.append("--continued")
    if task == "full":
        progressive_gate("two_objects_one_goal", recipe)
    stage = json.loads((ROOT / "configs/v3/task_stages.json").read_text())[task]
    name = f"bc_{task}_{recipe}80_s0"
    stages = [("collect_train", ["collect-stage", "--task", task, "--split", "train"]),
              ("collect_validation", ["collect-stage", "--task", task, "--split", "validation"]),
              ("train_s0", ["train", "--name", name, "--task", task, "--limit", "80", *extra]),
              ("tuning_s0", ["evaluate", "--name", name + "_tuning", "--first", str(stage["tuning_first"]),
                              "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"])]
    supervise(f"{task}_{recipe}", stages)
    rows = json.loads((ROOT / f"results/v3/validation/{name}_tuning.json").read_text())["rollouts"]
    if sum(r["physical_success"] for r in rows) < 12:
        print(f"{task}: seed-0 competence gate not reached; no matrix is launched", flush=True)
        return
    repeat = []
    for seed in (0, 1, 2):
        name = f"bc_{task}_{recipe}80_s{seed}"
        if seed:
            repeat.extend([
                (f"train_s{seed}", ["train", "--name", name, "--task", task, "--limit", "80", "--seed", str(seed), *extra]),
                (f"tuning_s{seed}", ["evaluate", "--name", name + "_tuning", "--first", str(stage["tuning_first"]),
                 "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]),
            ])
        repeat.append((f"confirmation_s{seed}", ["evaluate", "--name", name + "_confirmation",
                       "--first", str(stage["confirmation_first"]), "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]))
    supervise(f"{task}_{recipe}", repeat)
    progressive_gate(task, recipe)


def progressive_gate(task, recipe):
    evidence = []
    for seed in (0, 1, 2):
        for split in ("tuning", "confirmation"):
            name = f"bc_{task}_{recipe}80_s{seed}_{split}"
            path = ROOT / f"results/v3/validation/{name}.json"
            assert path.exists(), f"Progressive gate pending: {path.name}"
            result = json.loads(path.read_text())
            stage = json.loads((ROOT / "configs/v3/task_stages.json").read_text())[task]
            counts = validation_counts(result, seed, stage[split + "_first"])
            count = counts["physical"]
            evidence.append(dict(seed=seed, split=split, physical=count,
                                 strict=counts["strict"],
                                 checkpoint_sha256=result["identity"]["checkpoint_sha256"]))
            assert count >= 12, f"Progressive gate failed: {name}: {count}/20"
    write(ROOT / f"results/v3/{task}_competence_gate.json", evidence)
    return evidence


def two_object_control():
    """Bounded one-factor control after the rejected multi-object history pilot."""
    competence("bc")
    task, recipe = "two_objects_one_goal", "continued_state1"
    stage = json.loads((ROOT / "configs/v3/task_stages.json").read_text())[task]
    stages = []
    for seed in (0, 1, 2):
        name = f"bc_{task}_{recipe}80_s{seed}"
        stages.extend([
            (f"train_s{seed}", ["train", "--name", name, "--task", task, "--limit", "80",
             "--seed", str(seed), "--continued"]),
            (f"tuning_s{seed}", ["evaluate", "--name", name + "_tuning", "--first", str(stage["tuning_first"]),
             "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]),
        ])
        if seed == 0:
            supervise("two_object_history_control", stages)
            stages = []
            result = json.loads((ROOT / f"results/v3/validation/{name}_tuning.json").read_text())
            if validation_counts(result, 0, stage["tuning_first"])["physical"] < 12:
                print("History1 control failed competence gate; no repetition or complexity expansion", flush=True)
                return
        stages.append((f"confirmation_s{seed}", ["evaluate", "--name", name + "_confirmation",
                       "--first", str(stage["confirmation_first"]), "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]))
    supervise("two_object_history_control", stages)
    evidence = progressive_gate(task, recipe)
    write(ROOT / "configs/v3/two_objects_one_goal_selection.json",
          dict(recipe=recipe, continued=True, history=1, binary_gripper=False,
               evidence=evidence, reserved_test_used=False,
               decision="Three-seed physical competence on tuning and disjoint confirmation; only robot-state history differs from the rejected pilot"))


def routed_control():
    competence("bc")
    task, recipe = "two_objects_one_goal", "routed_continued_state1"
    stage = json.loads((ROOT / "configs/v3/task_stages.json").read_text())[task]
    stages = []
    for seed in (0, 1, 2):
        name = f"bc_{task}_{recipe}80_s{seed}"
        stages.extend([
            (f"train_s{seed}", ["train", "--name", name, "--model", "routed_bc", "--task", task,
             "--limit", "80", "--seed", str(seed), "--continued"]),
            (f"tuning_s{seed}", ["evaluate", "--name", name + "_tuning", "--first", str(stage["tuning_first"]),
             "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]),
        ])
        if seed == 0:
            supervise("two_object_routing_control", stages)
            stages = []
            result = json.loads((ROOT / f"results/v3/validation/{name}_tuning.json").read_text())
            if validation_counts(result, 0, stage["tuning_first"])["physical"] < 12:
                print("Routing pilot failed competence gate; no repetition or complexity expansion", flush=True)
                return
        stages.append((f"confirmation_s{seed}", ["evaluate", "--name", name + "_confirmation",
                       "--first", str(stage["confirmation_first"]), "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]))
    supervise("two_object_routing_control", stages)
    evidence = progressive_gate(task, recipe)
    write(ROOT / "configs/v3/two_objects_one_goal_selection.json",
          dict(recipe=recipe, model="routed_bc", continued=True, history=1, binary_gripper=False,
               evidence=evidence, reserved_test_used=False,
               decision="Three-seed physical competence on tuning and disjoint confirmation; sensor-moment routing is an explicit inductive bias"))


def interaction():
    competence("bc")
    stages = []
    for seed in (0, 1, 2):
        name = f"bc_continued_history480_s{seed}"
        stages.extend([
            (name + "_train", ["train", "--name", name, "--limit", "80", "--seed", str(seed),
             "--history", "4", "--continued"]),
            (name + "_tuning", ["evaluate", "--name", name + "_tuning",
             "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]),
        ])
    supervise("interaction", stages)


def interaction_confirmation():
    """Distinct validation scenes, declared before promoting the combined recipe."""
    stages = []
    for seed in (0, 1, 2):
        name = f"bc_continued_history480_s{seed}"
        stages.append((name + "_confirmation", ["evaluate", "--name", name + "_confirmation",
                       "--first", "110200", "--checkpoint", f"artifacts/v3/runs/{name}/best.pt"]))
    supervise("interaction_confirmation", stages)
    evidence = []
    for seed in (0, 1, 2):
        for recipe in ("original", "continued_history4"):
            name = f"bc_{recipe}80_s{seed}_confirmation"
            result = json.loads((ROOT / f"results/v3/validation/{name}.json").read_text())
            counts = validation_counts(result, seed, 110200)
            evidence.append(dict(recipe=recipe, seed=seed,
                                 **counts,
                                 checkpoint_sha256=result["identity"]["checkpoint_sha256"],
                                 result=f"results/v3/validation/{name}.json"))
    original = [r for r in evidence if r["recipe"] == "original"]
    candidate = [r for r in evidence if r["recipe"] == "continued_history4"]
    for seed in (0, 1, 2):
        originals = json.loads((ROOT / f"results/v3/validation/bc_original80_s{seed}_confirmation.json").read_text())["rollouts"]
        candidates = json.loads((ROOT / f"results/v3/validation/bc_continued_history480_s{seed}_confirmation.json").read_text())["rollouts"]
        assert all(a["initial_sensor_sha256"] == b["initial_sensor_sha256"] for a, b in zip(originals, candidates))
    accepted = (all(r["physical"] >= 12 and r["strict"] >= 12 for r in candidate)
                and sum(r["physical"] for r in candidate) >= sum(r["physical"] for r in original)
                and sum(r["strict"] for r in candidate) > sum(r["strict"] for r in original))
    selection = dict(recipe="continued_history4" if accepted else "original",
                     continued=accepted, history=4 if accepted else 1,
                     binary_gripper=False, evidence=evidence,
                     decision="Distinct validation only; each seed >=12/20 on both criteria, physical total no lower and strict total higher than original",
                     reserved_test_used=False)
    write(ROOT / "configs/v3/selection.json", selection)
    print(json.dumps(selection), flush=True)


def single_study():
    competence("bc")
    selection = json.loads((ROOT / "configs/v3/selection.json").read_text())
    assert selection["recipe"] == "continued_history4"
    stages = [("named_views_check", ["stage-checks", "--task", "single"])]
    for seed in (1,2):
        name = f"diffusion_original80_s{seed}"
        stages.extend([
            (name+"_train", ["train","--name",name,"--model","v1_diffusion","--limit","80","--seed",str(seed)]),
            (name+"_tuning", ["evaluate","--name",name+"_tuning","--checkpoint",f"artifacts/v3/runs/{name}/best.pt"]),
        ])
    for view in ("fusion","fixed","wrist"):
        for prior in (True,False):
            for seed in (0,1,2):
                name = f"single_{view}_{'prior' if prior else 'no_prior'}_s{seed}"
                checkpoint = f"artifacts/v3/runs/{name}/best.pt"
                if view == "fusion" and prior:
                    checkpoint = f"artifacts/v3/runs/bc_continued_history480_s{seed}/best.pt"
                else:
                    stages.append((name+"_train", ["train","--name",name,"--view",view,"--limit","80",
                                   "--seed",str(seed),"--continued","--history","4", *([] if prior else ["--no-prior"])]))
                stages.append((name+"_tuning", ["evaluate","--name",name+"_tuning","--checkpoint",checkpoint]))
    supervise("single_view_prior_study",stages)


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
