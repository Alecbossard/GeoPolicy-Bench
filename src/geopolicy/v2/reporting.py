"""Aggregate only executed rows; write an auditable before/after report and local demo."""

import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from geopolicy.io import save_json
from .config import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def cube(rows, condition, scenes):
    values = {
        (r["training_seed"], r["scene_seed"]): r["stable_success"]
        for r in rows
        if r["condition"] == condition
    }
    assert len(values) == 3 * len(scenes)
    return np.array([[values[seed, scene] for scene in scenes] for seed in [0, 1, 2]], float)


def bootstrap(delta):
    rng = np.random.default_rng(81)
    seed_index = rng.integers(3, size=(10000, 3))
    scene_index = rng.integers(delta.shape[1], size=(10000, delta.shape[1]))
    draws = delta[seed_index[:, :, None], scene_index[:, None, :]].mean((1, 2))
    return dict(
        difference_pp=float(delta.mean() * 100),
        interval95_pp=(np.percentile(draws, [2.5, 97.5]) * 100).tolist(),
    )


def markdown_table(headers, records):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in records)
    )


def report(recipe):
    from .failure_analysis import analyze_v1

    analyze_v1()
    plan = read("configs/v2/main_plan.json")["jobs"]
    e = recipe["evaluation"]
    scenes = list(
        range(e["reserved_test_first_seed"], e["reserved_test_first_seed"] + e["test_episodes"])
    )
    grouped = {}
    raw = []
    for job in plan:
        folder = f"{job['name']}_s{job['seed']}"
        rows = read(f"artifacts/v2/test/{folder}/rollouts.json")
        assert len(rows) == len(scenes) * len(e["conditions"])
        assert {r["checkpoint_sha256"] for r in rows} == {file_hash(job["checkpoint"])}
        grouped.setdefault(job["name"], []).extend(rows)
        raw += [dict(model=job["name"], **r) for r in rows]
    identities = {}
    for r in raw:
        identities.setdefault(r["scene_seed"], set()).add(
            r["initial_rgbd_robot_camera_pose_sha256"]
        )
    assert all(len(v) == 1 for v in identities.values()), "Unpaired final scene geometry"
    for name in ["wrist_prior", "wrist_no_prior"]:
        baseline = {
            (r["training_seed"], r["scene_seed"]): r
            for r in grouped[name]
            if r["condition"] == "nominal"
        }
        for row in grouped[name]:
            paired = baseline[row["training_seed"], row["scene_seed"]]
            assert (
                row["stable_success"] == paired["stable_success"]
                and row["steps"] == paired["steps"]
            )
            np.testing.assert_array_equal(row["first_action"], paired["first_action"])
    cells = []
    arrays = {}
    for name, rows in grouped.items():
        for condition in e["conditions"]:
            a = arrays[name, condition] = cube(rows, condition, scenes)
            selected = [r for r in rows if r["condition"] == condition]
            cells.append(
                dict(
                    model=name,
                    condition=condition,
                    successes=int(a.sum()),
                    episodes=a.size,
                    success_percent=float(a.mean() * 100),
                    seed_successes=a.sum(1).astype(int).tolist(),
                    instantaneous_successes=sum(r["instantaneous_success_v1"] for r in selected),
                    failure_stages=dict(Counter(r["failure_stage"] for r in selected)),
                    wrong_object_lifts=sum(r["wrong_object_lifted"] for r in selected),
                    collision_episodes=sum(r["collision"] for r in selected),
                    median_total_p95_ms=float(np.median([r["total_p95_ms"] for r in selected])),
                )
            )
    comparisons = []
    for condition in e["conditions"]:
        pairs = [("fusion_prior", "fixed_prior"), ("fusion_prior", "wrist_prior")]
        pairs += [(f"{view}_prior", f"{view}_no_prior") for view in ["fixed", "wrist", "fusion"]]
        for a, b in pairs:
            comparisons.append(
                dict(
                    a=a,
                    b=b,
                    condition=condition,
                    **bootstrap(arrays[a, condition] - arrays[b, condition]),
                )
            )
    before_after = []
    before_raw = []
    for mode, new in [("fusion", "fusion_prior"), ("act", "act_selected")]:
        old = []
        for seed in [0, 1, 2]:
            rows = read(f"artifacts/v2/before_after/{mode}_s{seed}/rollouts.json")
            old.extend(rows)
            before_raw += [dict(model=f"v1_{mode}", **r) for r in rows]
        b = cube(old, "nominal", scenes)
        a = arrays[new, "nominal"]
        before_after.append(
            dict(
                model=mode,
                v1_successes=int(b.sum()),
                v2_successes=int(a.sum()),
                episodes=a.size,
                v1_seed_successes=b.sum(1).astype(int).tolist(),
                v2_seed_successes=a.sum(1).astype(int).tolist(),
                **bootstrap(a - b),
            )
        )
    counterfactual = []
    cf_raw = []
    geometry = {}
    for name in ["fusion_prior", "fusion_no_prior", "act_selected"]:
        all_four = []
        seed_results = []
        for seed in [0, 1, 2]:
            records = []
            for obj in [0, 1]:
                for goal in [0, 1]:
                    rows = read(
                        f"artifacts/v2/counterfactual/{name}_s{seed}_o{obj}_g{goal}/rollouts.json"
                    )
                    assert len(rows) == e["counterfactual_scenes"]
                    for row in rows:
                        geometry.setdefault(row["scene_seed"], set()).add(
                            row["initial_rgbd_robot_camera_pose_sha256"]
                        )
                    records.extend(rows)
                    cf_raw += [dict(model=name, **r) for r in rows]
            grouped_cf = {}
            for row in records:
                grouped_cf.setdefault(row["scene_seed"], []).append(row["stable_success"])
            assert all(len(v) == 4 for v in grouped_cf.values())
            count = sum(all(v) for v in grouped_cf.values())
            all_four.append(count)
            seed_results.append(sum(r["stable_success"] for r in records))
        counterfactual.append(
            dict(
                model=name,
                successes_all_four=sum(all_four),
                scene_seed_pairs=3 * e["counterfactual_scenes"],
                per_training_seed_all_four=all_four,
                individual_successes=sum(seed_results),
                individual_episodes=12 * e["counterfactual_scenes"],
            )
        )
    assert all(
        len(v) == 1 for v in geometry.values()
    ), "Counterfactual sensor inputs must be identical"
    resources = []
    models = []
    for job in plan:
        root = Path(job["checkpoint"]).parent
        manifest = read(root / "manifest.json")
        models.append(
            dict(
                model=job["name"],
                seed=job["seed"],
                parameters=manifest["parameters"],
                optimization_seconds=manifest["optimization_seconds"],
                peak_allocated_bytes=manifest["peak_allocated_bytes"],
            )
        )
        resources.extend(read(root / "resources.json"))
    resource_summary = dict(
        maximum_gpu_used_mib=max(r["gpu_used_free_mib_temperature_c"][0] for r in resources),
        minimum_free_gpu_mib=min(r["gpu_used_free_mib_temperature_c"][1] for r in resources),
        maximum_temperature_c=max(r["gpu_used_free_mib_temperature_c"][2] for r in resources),
        maximum_process_private_gib=max(r["private_bytes"] for r in resources) / 1024**3,
        minimum_free_commit_gib=min(r["free_commit_bytes"] for r in resources) / 1024**3,
        minimum_free_disk_gib=min(r["free_disk_bytes"] for r in resources) / 1024**3,
        summed_main_optimization_seconds=sum(m["optimization_seconds"] for m in models),
        maximum_torch_allocated_mib=max(m["peak_allocated_bytes"] for m in models) / 1024**2,
    )
    selection = read("results/v2/selection.json")
    control = read("artifacts/v2/pilot_evaluations/diffusion_original_data/rollouts.json")
    suffix = read("artifacts/v2/pilot_evaluations/diffusion_suffix_only/rollouts.json")
    summary = dict(
        cells=cells,
        comparisons=comparisons,
        before_after=before_after,
        counterfactual=counterfactual,
        resources=resource_summary,
        models=models,
        selected=selection,
        rollout_counts=dict(
            main=len(raw), before_after=len(before_raw), counterfactual=len(cf_raw)
        ),
        paired_main_initial_sensors=True,
        wrist_only_outcomes_unchanged_by_fixed_corruption=True,
        identical_counterfactual_sensors=True,
        suffix_control=dict(
            original_successes=sum(r["stable_success"] for r in control),
            extended_successes=sum(r["stable_success"] for r in suffix),
            episodes=20,
        ),
        protocol_sha256=file_hash("configs/v2/final_protocol.json"),
    )
    save_json("results/v2/summary.json", summary)
    for name, records in [
        ("main_rollouts", raw),
        ("before_after_rollouts", before_raw),
        ("counterfactual_rollouts", cf_raw),
    ]:
        save_json(f"results/v2/raw/{name}.json", records)
        with Path(f"results/v2/raw/{name}.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for r in records for k in r}))
            writer.writeheader()
            writer.writerows(records)
    validation_raw = []
    for group, parent in [
        ("diagnostic", "diagnostics"),
        ("privileged_component_diagnostic", "act_oracle"),
        ("pilot", "pilot_evaluations"),
        ("confirmation", "confirmation"),
    ]:
        for path in sorted(Path(f"artifacts/v2/{parent}").glob("*/rollouts.json")):
            validation_raw.extend(
                dict(group=group, model=path.parent.name, **row) for row in read(path)
            )
    for parent in ["pilot_evaluations", "confirmation"]:
        for path in sorted(Path(f"artifacts/v2/intermediate_v2a/{parent}").glob("*/rollouts.json")):
            validation_raw.extend(
                dict(group="intermediate_v2a_" + parent, model=path.parent.name, **row)
                for row in read(path)
            )
    actor_initial = Path("artifacts/v2/initial_act_confirmation/rollouts/rollouts.json")
    if actor_initial.exists():
        validation_raw.extend(
            dict(group="initial_act_confirmation", model="act_history_binary", **row)
            for row in read(actor_initial)
        )
    save_json("results/v2/raw/validation_rollouts.json", validation_raw)
    with Path("results/v2/raw/validation_rollouts.csv").open(
        "w", newline="", encoding="utf8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in validation_raw for k in r}))
        writer.writeheader()
        writer.writerows(validation_raw)
    save_json(
        "results/v2/training_manifests.json",
        {
            f"{j['name']}_s{j['seed']}": read(Path(j["checkpoint"]).parent / "manifest.json")
            for j in plan
        },
    )
    save_json(
        "results/v2/learning_curves.json",
        {
            f"{j['name']}_s{j['seed']}": read(Path(j["checkpoint"]).parent / "learning_curve.json")
            for j in plan
        },
    )
    save_json(
        "results/v2/resource_gates.json",
        {
            p.parent.name: read(p)
            for p in sorted(Path("artifacts/v2/gates").glob("*/gate_report.json"))
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(grouped)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = ["#64748b", "#cbd5e1", "#0f766e", "#99f6e4", "#7c3aed", "#ddd6fe", "#d97706"]
    for ax, condition in zip(axes, e["conditions"]):
        for i, name in enumerate(names):
            a = arrays[name, condition]
            ax.bar(i, a.mean() * 100, color=colors[i])
            ax.scatter([i] * 3, a.mean(1) * 100, c="black", s=14, zorder=3)
        ax.set(title=condition, xticks=range(len(names)), xticklabels=names, ylim=(0, 100))
        ax.tick_params(axis="x", labelrotation=75)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Stable placement (%)")
    fig.suptitle("Equal data and budgets; dots = 3 training seeds; 50 shared scenes per condition")
    fig.tight_layout()
    Path("results/v2/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/v2/figures/main_comparison.png", dpi=160)
    plt.close(fig)
    from .demo import build_demo

    demo = build_demo(recipe)
    clean = Path(".venv-repro/Scripts/python.exe")
    assert clean.exists(), "Previously pinned clean reproduction environment missing"
    clean_tests = subprocess.run(
        [str(clean), "-m", "pytest", "tests/v2", "-q"], capture_output=True, text=True
    )
    assert clean_tests.returncode == 0, clean_tests.stdout + clean_tests.stderr
    clean_demo = subprocess.run(
        [str(clean), "-m", "geopolicy.v2", "demo", "--out", "artifacts/v2/clean_demo"],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert clean_demo.returncode == 0, clean_demo.stdout + clean_demo.stderr
    clean_report = dict(
        tests=clean_tests.stdout,
        demo=read("artifacts/v2/clean_demo/verification.json"),
        executable=str(clean),
        checkpoint_hash_verified=True,
    )
    save_json("results/v2/clean_reproduction.json", clean_report)
    write_documents(recipe, summary, demo)
    from .preservation import preserve_v1

    preserve_v1()
    print(json.dumps(summary["rollout_counts"], indent=2), flush=True)
    return summary


def write_documents(recipe, summary, demo):
    cells = summary["cells"]
    comparisons = summary["comparisons"]
    before = summary["before_after"]
    table = markdown_table(
        ["Model", "Condition", "Stable /150", "Seeds (success /50)", "Ever V1 instant /150"],
        [
            (
                c["model"],
                c["condition"],
                f"{c['successes']}/150 ({c['success_percent']:.1f}%)",
                str(c["seed_successes"]),
                c["instantaneous_successes"],
            )
            for c in cells
        ],
    )
    compare = markdown_table(
        ["Paired comparison", "Condition", "Difference pp", "Descriptive 95% interval"],
        [
            (
                f"{c['a']} − {c['b']}",
                c["condition"],
                f"{c['difference_pp']:+.1f}",
                f"[{c['interval95_pp'][0]:+.1f}, {c['interval95_pp'][1]:+.1f}]",
            )
            for c in comparisons
        ],
    )
    before_table = markdown_table(
        ["Model", "V1 stable /150", "V2 stable /150", "Paired difference pp [95%]"],
        [
            (
                c["model"],
                c["v1_successes"],
                c["v2_successes"],
                f"{c['difference_pp']:+.1f} [{c['interval95_pp'][0]:+.1f}, {c['interval95_pp'][1]:+.1f}]",
            )
            for c in before
        ],
    )
    failure_table = markdown_table(
        [
            "Model",
            "Condition",
            "Selection/no approach",
            "Grasp",
            "Transport",
            "Release",
            "Post-release criterion",
        ],
        [
            (
                c["model"],
                c["condition"],
                c["failure_stages"].get("selection", 0)
                + c["failure_stages"].get("selection_no_approach", 0),
                c["failure_stages"].get("grasp", 0),
                c["failure_stages"].get("transport", 0),
                c["failure_stages"].get("release", 0),
                c["failure_stages"].get("unstable_placement", 0),
            )
            for c in cells
        ],
    )
    selection = summary["selected"]
    pilot_table = markdown_table(
        ["Pilot", "Stable /20", "Wrong lift", "Collision episodes", "Correct transport"],
        [
            (
                name,
                row["stable_successes"],
                row["wrong_object_lifts"],
                row["collision_episodes"],
                row["transport_reached"],
            )
            for name, row in selection["pilot_scores"].items()
        ],
    )
    diagnostics = read("results/v2/diagnostics.json")
    oracle = read("results/v2/act_component_interventions.json")
    oracle_table = markdown_table(
        ["Validation ACT control", "Stable /20", "Meaning"],
        [
            (
                "Unassisted preserved ACT",
                oracle["unassisted_stable_successes"],
                "Learned controller",
            ),
            (
                "Scripted privileged motion + ACT gripper",
                oracle["motion_oracle_stable_successes"],
                "Diagnostic intervention only",
            ),
            (
                "ACT motion + scripted privileged gripper",
                oracle["gripper_oracle_stable_successes"],
                "Diagnostic intervention only",
            ),
        ],
    )
    prior = diagnostics["act_recorded_validation"]["prior"]
    posterior = diagnostics["act_recorded_validation"]["posterior_oracle"]
    confirm = {
        family: read(f"artifacts/v2/confirmation/{family}/rollouts.json")
        for family in ["act", "diffusion"]
    }
    confirm_text = ", ".join(
        f"{name}: {sum(r['stable_success'] for r in rows)}/20" for name, rows in confirm.items()
    )
    suffix = summary["suffix_control"]
    cf_table = markdown_table(
        ["Model", "All four /30 scene-seed pairs", "Per training seed", "Individual /120"],
        [
            (
                c["model"],
                c["successes_all_four"],
                c["per_training_seed_all_four"],
                c["individual_successes"],
            )
            for c in summary["counterfactual"]
        ],
    )
    figures = "![Three-seed stable placement](../results/v2/figures/main_comparison.png)"
    text = f"""# V2 measured before/after report

All results below were actually executed locally. V1 is retained at tag `geopolicy-v1-2026-09-30` (542f132); its original results, weights, source and historical report are unchanged. The new evaluation uses a stricter stable-placement criterion and longer 240-step horizon, so historical instantaneous rates are contextual figures. The before/after table re-evaluates preserved V1 checkpoints on exactly the same new scenes and metric.

## Question and controls

Does calibrated fixed+wrist RGB-D fusion help beyond either view alone, and how much does the manual color prior contribute? Six diffusion variants cross fixed/wrist/fusion with prior present/absent. All use the same 200 training episodes extended with 30 actual PPO-teacher steps after the original end, the same 21 validation episodes, frozen V1 normalization, 7D OSC actions, 512 points, 8-step prediction, 2-step execution, 8,000 updates, batch 32 and three training seeds. The prior ablation removes the additive chromatic attention bias; it retains learned attention and explicit XYZRGB moments. RGB still contains colors. The models are compact independent adaptations, without pretrained vision.

The selected ACT uses `{selection['selected']['act']}`: a compact random-CNN RGB action-chunk Transformer, trained directly on its zero-latent deployment path, with the selected proprioceptive history and gripper output. Its modality and capacity differ from diffusion; its equal optimizer budget does not establish architectural parity with original ACT. The rejected `act_point` was an explicit 3D adaptation, not an official RGB ACT reproduction. PPO and SmolVLA remain the unchanged V1 extensions; no new improvement is claimed for them.

## Diagnoses before increasing budgets

Recorded versus live first-frame ACT RGB and normalized state match exactly on scene 100000. Action normalization roundtrip max error is {diagnostics['action_normalization_roundtrip_max_error']:.3g}. The physical Panda probe verified negative command opens and positive command closes; the backend uses command sign, so a small positive value is not a weak close. The new binary output maps to physical ±1 after normalization, tested with a nonzero action mean. Live history copies robot state arrays at observation time to avoid MuJoCo buffer aliasing and uses no future state or privileged object pose.

V1 ACT on recorded validation: normalized L1 prior {prior['normalized_l1']:.6f}, future-action-conditioned posterior oracle {posterior['normalized_l1']:.6f}; gripper sign accuracy {prior['gripper_sign_accuracy']*100:.2f}%. The tiny posterior/prior difference does not demonstrate a latent train/inference mismatch causing the failures. Zero-latent inference is also used in [original ACT](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py) and [LeRobot ACT 0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/modeling_act.py). Correct per-frame preprocessing and low offline error do not establish closed-loop competence.

The first-frame input sensitivity interventions are in `results/v2/act_input_sensitivity.json`. Image permutation and instruction flips change the output locally, but this does not prove grounded task behavior or identify a unique cause. Grasp and selection dominate the fresh V1 validation failures. V1 ACT's historical 0/60 nominal test remains unchanged; it achieved 2/20 on the new stable-metric diagnostic validation set, so '0%' is not an intrinsic impossibility claim. The scripted diagnostic reference achieved 20/20 stable placements; students receive none of its truth-pose waypoints.

{oracle_table}

These paired component interventions use the same 100200–100219 validation scenes. Replacing the six motion commands raises the score from {oracle['unassisted_stable_successes']}/20 to {oracle['motion_oracle_stable_successes']}/20; replacing only the gripper gives {oracle['gripper_oracle_stable_successes']}/20. This localizes a substantial bottleneck to the learned motion-command path under these interventions, rather than establishing a gripper-sign bug. The script uses ground-truth object/goal waypoints and privileged phase timing, so the intervention does not isolate perception, temporal prediction or architectural capacity as a unique cause. No oracle output counts as learned-baseline improvement, no final-test oracle is allowed, and these diagnostic results do not retune the frozen main choices. The unassisted score and stage traces remain the baseline evidence.

The original demonstrations terminate at the instantaneous metric. All 221 exact-prefix trajectories were continued for 30 actual selected-PPO-teacher steps; all extended teacher trajectories achieved sustained stability. This demonstrates that recording covers post-release behavior. In a matched 3,000-update seed-0 control, original data gave {suffix['original_successes']}/20 stable placements versus {suffix['extended_successes']}/20 after continuation. This result does not support continuation as a performance improvement. The original-data control was outside the preset ranking: the final view/prior study intentionally uses one shared extended dataset to cover the new scoring interval, rather than claiming it was the best dataset on validation. That design choice and possible degradation from the added data are limitations; the full V2 before/after comparison must establish the resulting behavior. This small single-seed control neither isolates every mechanism nor demonstrates a general gain. All negative pilots and source data are retained.

## Intermediate implementation correction

An intermediate V2 binary-gripper variant left the seventh diffusion output unsupervised while still using it as part of the iterative DDIM state. On the same diagnostic batch its seventh-output gradient L1 was 0 without auxiliary supervision and 7.00354 with it. This is a demonstrated structural defect introduced in V2's intermediate adaptation, not in preserved V1. Its contribution to manipulation failures was not isolated causally. A 0.1-weight normalized-gripper MSE now trains that internal channel while the separate classifier still provides the physical sign command. Seven targeted tests cover this, rotated-corner containment and the existing contracts.

Before any reserved test, ten completed intermediate 8,000-update runs and one partial checkpoint, plus all pilots/initial confirmation, were archived separately in `artifacts/v2/intermediate_v2a`. Their negative results are retained in `results/v2/intermediate_v2a_summary.json` and the validation raw table. Pilot selection was repeated after the correction using the validation set; final independent confirmation uses 100280–100299 for diffusion and 100320–100339 for ACT. Final budgets remain 8,000 updates per model. The additional compute is recorded as intermediate diagnostic work, not silently counted as one training run or used to select on the test.

## Validation-only selection

Six initial corrected 3,000-update pilots used the same optimizer and data budget. A seventh RGB ACT pilot adaptively broadcasts the learned instruction embedding to all visual tokens, motivated by the recorded first-frame sensitivity probe. It adds no parameters and keeps the same data/update budget; its 0/20 stable result rejects this improvement on validation. The original ranking rule remained unchanged. Selection rule was recorded before comparative results: most stable successes, then fewer wrong-object lifts, fewer collisions, more correct transport, then declared order. Selected diffusion: `{selection['selected']['diffusion']}`; selected ACT: `{selection['selected']['act']}`. Disjoint confirmation: diffusion 100280–100299, ACT 100320–100339, reported without retuning: {confirm_text}.

{pilot_table}

The additional ACT diagnostic is specified in `configs/v2/act_diagnostic.json`; earlier ACT confirmation is retained separately. Main choices, hashes, budgets and source identities were frozen in `configs/v2/final_protocol.json` before any new reserved test. Final scenes 300000–300049 are distinct from the V1 test and all validation scenes. Results cannot change the selected recipe. Three training seeds are crossed with 50 identical scenes per condition. Bootstrap resamples training seeds and paired scene columns jointly (10,000 draws, RNG 81); percentile intervals are descriptive, with only three training seeds, no multiplicity correction, and no claim about unseen scene families. An all-zero observed sample does not establish a zero population probability.

## New reserved test

{table}

{figures}

{compare}

The protocol marks four primary comparisons; other view/prior differences are exploratory. A positive point estimate alone is not proof of an advantage. The wrist-only camera-loss cell is a control for the surviving sensor; it helps distinguish fusion from merely retaining that view. Manual prior effects are separated within each view under an otherwise common recipe.

## Fair before/after

{before_table}

The full V2 recipe bundles continued data, causal history, loss/output changes and selected training paths. These comparisons demonstrate the resulting behavior in this scene family; they do not attribute a gain to one component. V1's historical nominal fusion 30.7%, fixed 30.3%, missing-fixed fusion 17.0% versus fixed 1.0% used instantaneous success and a different test set. They remain documented in the untouched `docs/final_report.md` and cannot be substituted for this stable-score comparison.

## Failure localization and instruction controls

{failure_table}

Stages are evaluator-only geometric/contact proxies: selection (wrong object lifted without selected lift, or no target approach), grasp (no selected lift), transport (no correct destination reach), release (no full contained open-finger release), unstable (release seen but no continuous stable interval). They are mutually exclusive failure labels, not a claim that the teacher phase machine was inferred. The JSON label `unstable_placement` means the complete post-release criterion was not held for one second: it can reflect hand closure even when the cube itself remains still. It is not proof of physical object instability; positions, speeds and contacts in the trace distinguish these cases. Collision and wrong-target flags remain separate; success is not certified collision-free.

{cf_table}

All four instructions reuse identical RGB, depth, camera poses and robot state at initialization; sensor hashes were verified across views/seeds and instruction changes. Instruction tokens change; no object truth enters student inputs. Strict all-four completion and individual rates constrain language-grounding claims. Tokens encode only four known combinations; this is not open-vocabulary language understanding.

## Stable placement contract

At 20 Hz, valid samples must span one continuous second (at least 21 samples). The selected cube must previously lift above 0.89 m, have center height 0.816–0.855 m and all rotated XY corners contained in the chosen tray's inner half-extents [0.063, 0.060] m minus 1 mm margin. Fingers must be open at least 45 mm with no selected-object finger contact, object linear speed ≤0.02 m/s and angular speed ≤0.25 rad/s. Invalid samples reset the interval. This primary protocol also requires fingers to remain open throughout the dwell; closing empty fingers after withdrawing can therefore fail the strict score despite an otherwise stable object. This conservative robot-ending-posture requirement is explicit, and raw traces retain object pose/speed and hand width so it is not misattributed to object motion. The policy keeps acting throughout verification, with no assisted release. Evaluation ends at stable success or 240 steps. The tray dimensions are specific to this simulator asset; the criterion measures a one-second dwell, not indefinite stability or general collision safety. Unit tests cover interruption, transient crossing, contact, finger width, speed, containment and prior lift.

## Execution, verification and reproduction

One GPU job at a time on RTX 4060 Laptop 8 GB; all optimizations FP32. Per-preset actual AdamW/dual-render memory gates and exact next-update recovery passed. Main training resource summary: `{json.dumps(summary['resources'], ensure_ascii=False)}`. CUDA allocation excludes driver/context/renderer memory; device-wide and process committed memory are logged separately. Training gates monitor temperature, disk, GPU and Windows commit headroom, preserve periodic checkpoints, and stop on persistent unsafe samples. Main and evaluation manifests contain timings and resource traces; wall time includes local hardware effects.

Executed final counts: {summary['rollout_counts']['main']} main, {summary['rollout_counts']['before_after']} fair before/after, {summary['rollout_counts']['counterfactual']} counterfactual rollouts. Raw CSV/JSON live in `results/v2/raw`; per-step traces and checkpoints remain in local ignored `artifacts/v2`. Before opening final test, all original 274 HDF5 hashes, 42 original checkpoints and 122 tracked V1 files passed checks; all 221 augmented prefixes are array-exact. Continuing the real fusion checkpoint for its last 500 updates reproduced model, optimizer, scheduler, EMA and configuration exactly. Test outputs are hash-bound to frozen checkpoints and source. See `results/v2/pretest_verification.json` and `results/v2/v1_preservation.json`.

The compact local demo bundle contains only an EMA policy, recipe, hashes and raw expected examples. It needs no training data or pretrained-model download. The earliest success and earliest failure of predeclared fusion-prior seed 0 are shown, when present; this selection is disclosed and is not aggregate evidence. A separate pinned `.venv-repro` passed V2 tests and replayed the compact checkpoint with exact first actions, steps, initial sensor hashes and success labels. See `results/v2/clean_reproduction.json` and [reproduction instructions](v2_reproduction.md).

## Limits and defensible claims

Absolute manipulation reliability remains the limiting factor. Three seeds and 50 scenes cannot establish broad robot-learning superiority. Simulation uses known calibration, rigid colored cubes, four semantic tokens, selected successful demonstrations and a privileged phase-based BC-initialized PPO teacher. No real robot, Isaac, Jetson, open-vocabulary instruction or sim-to-real transfer was executed. Hardware timings apply to this PC. Negative ACT/3D pilots and failed V1 PPO/SmolVLA runs remain visible. There is no online publication or CV modification.
"""
    Path("docs/v2_report.md").write_text(text, encoding="utf8")
    nominal = next(c for c in cells if c["model"] == "fusion_prior" and c["condition"] == "nominal")
    loss = next(
        c
        for c in comparisons
        if c["a"] == "fusion_prior"
        and c["b"] == "fixed_prior"
        and c["condition"] == "fixed_camera_missing"
    )
    bullet = f"• Conçu un benchmark de manipulation Panda sous MuJoCo/robosuite : comparaison RGB-D fixe/poignet/fusion et ablation du prior couleur (21 modèles, trois seeds, {summary['rollout_counts']['main']:,} rollouts réservés) ; placement stable testé, reprise exacte des checkpoints et démonstration locale reproduite.\n"
    Path("docs/v2_cv_bullet.txt").write_text(bullet, encoding="utf8")
    chosen_success = next((r for r in demo["expected_rollouts"] if r["stable_success"]), None)
    chosen_failure = next((r for r in demo["expected_rollouts"] if not r["stable_success"]), None)
    videos = []
    for label, record in [("success", chosen_success), ("failure", chosen_failure)]:
        if record:
            filename = f"nominal_{record['scene_seed']}_{'stable' if record['stable_success'] else 'failure'}.mp4"
            videos.append(f"[{label.capitalize()} video](results/v2/demo/{label}/{filename})")
    video_links = " · ".join(videos)
    fusion_before = next(c for c in before if c["model"] == "fusion")
    act_nominal = next(
        c for c in cells if c["model"] == "act_selected" and c["condition"] == "nominal"
    )
    readme = f"""# GeoPolicy-Bench

**Question:** does calibrated fixed+wrist RGB-D fusion improve instruction-conditioned Panda placement beyond either camera alone, and how much comes from a manual color prior?

**Measured V2:** fusion achieved **{nominal['successes']}/150 ({nominal['success_percent']:.1f}%) stable nominal placements** over three training seeds and 50 reserved scenes. Under missing fixed camera, fusion minus fixed was **{loss['difference_pp']:+.1f} pp**, descriptive 95% interval **[{loss['interval95_pp'][0]:+.1f}, {loss['interval95_pp'][1]:+.1f}]**. This is a compact simulation study with limited reliability; a positive point estimate alone does not establish superiority. The selected compact ACT achieved {act_nominal['successes']}/150 stable nominal placements.

{video_links} — actual disclosed test episodes, local checkpoint replay verified in a separate pinned environment.

![Stable placement comparison](results/v2/figures/main_comparison.png)

## Reproduce the short local demo

Validated on Windows, Python 3.11, RTX 4060 Laptop 8 GB. From the project root:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\\Scripts\\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\\Scripts\\python.exe -r requirements-lock.txt
uv pip install --python .venv\\Scripts\\python.exe --no-deps -e .
# Restore/extract the local artifacts/v2/demo_bundle.zip into artifacts/v2/demo
.venv\\Scripts\\python -m geopolicy.v2 demo
```

No training data is needed for the demo. Weights stay local under ignored `artifacts/`; the Git repository alone does not contain them. Raw expected actions/metrics and hashes accompany the local bundle. [Full commands and exact artifact requirements](docs/v2_reproduction.md), [measured before/after report](docs/v2_report.md), [raw test CSV](results/v2/raw/main_rollouts.csv), [configuration](configs/v2/recipe.json), [progress](PROGRESS.md).

## What is controlled

Six diffusion variants: fixed, wrist and fusion, each with/without the chromatic prior. Same 200 successful demonstration prefixes plus 30 recorded PPO-teacher steps after release, 21 validation episodes, frozen normalization, 7D actions, 512 points, 8,000 updates, batch 32, three training seeds. Seven targeted 3,000-update pilots select the recipe on validation; a new frozen test uses 50 shared scenes in nominal, fixed occlusion and fixed-camera-loss conditions. ACT is a compact RGB adaptation with different capacity; a rejected 3D ACT-style pilot is reported separately.

The strict success metric requires full contained placement, low object speed, and fingers kept open without finger/object contact for one continuous second while the policy keeps acting. Closing the empty hand during that dwell can fail this conservative score; such a failure alone does not prove the object moved. Failure traces distinguish target selection, grasp, transport, release and post-release stability. Normalization, rendering alignment, physical gripper sign and real checkpoint continuation were verified. Per-frame loss is not evidence of manipulation success.

## Before/after and limits

On the same new nominal test and stable criterion, preserved V1 fusion scored {fusion_before['v1_successes']}/150 versus V2 {fusion_before['v2_successes']}/150. The V2 recipe changes multiple components; this does not identify one causal improvement. V1 historical instantaneous scores remain intact at tag `geopolicy-v1-2026-09-30`, with its [original README](docs/v1/README_original.md) and [original report](docs/final_report.md). All new experiments are separate under `artifacts/v2` and `results/v2`.

Known calibrated simulation, two colored cubes, two receptacles and four semantic instruction tokens; no pretrained vision, real-robot transfer, open-vocabulary grounding or official ACT/DP3 reproduction claimed. A scripted phase curriculum initializes a real PPO teacher; its historical PPO update showed no success gain over BC. PPO and SmolVLA remain secondary V1 extensions. Three seeds, narrow scenes, negative outcomes and residual failures limit the conclusions. Nothing was published online and the user's CV was not edited.

[Architecture and sources](docs/architecture.md) · [V2 design](docs/v2_design.md) · [Third-party notices](docs/THIRD_PARTY_NOTICES.md) · [Factual CV bullet proposal](docs/v2_cv_bullet.txt)
"""
    Path("README.md").write_text(readme, encoding="utf8")
