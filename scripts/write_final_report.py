"""Render the final report from completed, measured artifacts; no estimated scores."""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def percent(value):
    return f"{100 * value:.1f}%"


def gib(value):
    return f"{value / 1024**3:.3f}"


primary = read("results/benchmark_summary.json")
secondary = read("results/secondary_summary.json")
rollouts = read("results/raw/student_rollouts.json")
teacher = read("results/raw/final_teacher_controls.json")
teacher_audit = read("results/teacher_update_audit.json")
teacher_resource = read("results/teacher_resource_profile.json")
training = read("results/training_manifests.json")
checkpoint_audit = read("results/checkpoint_audit.json")
selected_updates = {r["run"]: r["selected_update"] for r in checkpoint_audit["main_runs"]}
vla_train = read("artifacts/smolvla_s0/manifest.json")
onnx = read("results/onnx_report.json")
reproduction = read("results/clean_reproduction.json")
full_resume = read("results/full_training_resume.json")
assert full_resume["all_exact"]
native_export = read("artifacts/lerobot_dataset/export_report.json")
compact_resources = read("results/combined_compact_resources.json")
vla_resources = read("results/combined_vla_resources.json")
stages = read("artifacts/finalization_stages.json")
assert len(rollouts) == 2340 and len(primary["per_seed"]) == 45
assert all(
    stages.get(name, {}).get("complete")
    for name in [
        "combined_compact",
        "combined_vla",
        "gpu_resume",
        "clean_tests",
        "secondary_summary",
    ]
)
assert native_export["reload_source_state_action_rgb_exact"]
assert reproduction["steps_match"] and reproduction["first_action_max_abs_error"] == 0
assert all(
    r["optimizer_states_loaded_and_live"] and r["dual_rgbd_renderer_live"]
    for r in compact_resources + vla_resources
)

condition_names = {
    "nominal": "Nominal",
    "occlusion": "Fixed RGB-D occlusion",
    "depth_degraded": "Depth noise / holes",
    "fixed_camera_missing": "Fixed view missing",
    "extrinsic_error": "Fixed extrinsic error",
}
conditions = list(condition_names)
aggregate = {(r["mode"], r["condition"]): r for r in primary["aggregate"]}
comparison = {r["condition"]: r for r in primary["paired_fusion_comparison"]}
nominal = comparison["nominal"]
lo, hi = nominal["paired_crossed_bootstrap95_pp"]
evidence = (
    "The descriptive interval excludes zero in this scene family."
    if lo > 0 or hi < 0
    else "The descriptive interval includes zero; these runs do not establish a consistent nominal advantage."
)
lines = [
    "# GeoPolicy Bench — measured final report",
    f"\nGenerated {datetime.now().isoformat(timespec='seconds')} from completed local artifacts.",
    "\nThe experimental core ran on one RTX 4060 Laptop 8 GB / Ryzen 7 7435HS / 16 GB RAM Windows PC. "
    "Nine compact models were trained and evaluated in 2,340 primary closed-loop rollouts. "
    "A real pretrained SmolVLA model was additionally fine-tuned and tested. "
    "No GitHub/Hub publication or CV edit was performed.",
    f"\nNominal fusion success is {percent(aggregate['fusion', 'nominal']['mean_success_rate'])}, "
    f"versus {percent(aggregate['mono', 'nominal']['mean_success_rate'])} for the same monocular architecture. "
    f"The paired difference is {nominal['fusion_minus_mono_pp']:+.1f} percentage points "
    f"(descriptive crossed-bootstrap 95% interval [{lo:.1f}, {hi:.1f}]). {evidence}",
    "\n## Protocol and scope",
    "\nThe protocol was frozen before principal training and final test. Checkpoints were selected "
    "by fixed offline validation loss, never test success. Mono/fusion share the same 200 successful "
    "demonstrations, normalization, 1,088,455-parameter architecture, 8,000 AdamW updates, batch 32, "
    "horizon 8, execute 2, 512-point budget and RGB instruction prior. Only the input views differ. "
    "Three training seeds are paired on identical test scene IDs. ACT uses the same demonstrations "
    "and update budget but different modalities and architecture. SmolVLA is exploratory and has a different budget.",
    "\n"
    + table(
        ["Policy", "Parameters", "Inputs", "Train seeds / updates / batch", "Pretraining"],
        [
            [
                "Mono diffusion",
                "1,088,455",
                "Fixed XYZRGB + robot state + explicit semantic tokens",
                "3 / 8,000 / 32",
                "None",
            ],
            [
                "Fusion diffusion",
                "1,088,455",
                "Fixed + wrist XYZRGB, shared 512 points; same conditioning",
                "3 / 8,000 / 32",
                "None",
            ],
            [
                "Compact ACT adaptation",
                "1,051,239",
                "Two RGB64 + robot state + semantic tokens",
                "3 / 8,000 / 32",
                "None",
            ],
            [
                "SmolVLA",
                "450,046,176 total; 99,880,992 trainable",
                "Two RGB128 resized/padded512 + robot state + text",
                "1 / 500 / 1, accumulation4",
                "Pinned authentic base; VLM frozen",
            ],
        ],
    ),
    "\nThe custom task is Panda selection and placement with two known cube colors, two receptacle "
    "colors, positional/color-slot randomization and a distractor. Successful placement requires "
    "the instructed cube to have previously lifted, lie within 4.3 cm XY of the chosen tray, be at "
    "the accepted height and be released. This is an instantaneous condition, not a sustained "
    "stability test. Wrong target, wrong destination, drops and contacts are separately recorded. "
    "Collision denotes robot/gripper contact penetrating table/tray-wall/distractor by over1mm. "
    "Students receive only RGB-D, calibration, allowed robot proprioception and instruction. "
    "Object truth, segmentation and teacher phases are excluded from student batches.",
    "\nThe engineered RGB chromatic attention prior only covers four known color meanings. "
    "This is an independent DP3/ACT-inspired adaptation, not an exact paper reproduction. "
    "The score is not LIBERO, arbitrary-language grounding or physical robot performance.",
    "\n## PPO expert, controls and data provenance",
    "\nThe teacher uses a privileged scripted eight-phase waypoint curriculum. Its actor was "
    "initialized with 2,500 supervised updates on 64 clearly labeled scripted bootstrap episodes, "
    "then actually optimized with SB3 PPO for 2,048 on-policy transitions. Task sequencing remains "
    "scripted. Selected teacher demonstrations are rollouts of that PPO checkpoint, not the bootstrap script.",
    "\n"
    + table(
        ["Control", "Final successful /100", "Test scenes"],
        [
            [name, f"{sum(r['success'] for r in rows)}/{len(rows)}", "200000–200099"]
            for name, rows in teacher.items()
        ],
    ),
    f"\nThe selected actor changed by L2={teacher_audit['updates'][0]['actor_parameter_delta_l2_from_bc']:.6f} "
    f"from BC; critic/other parameters changed by {teacher_audit['updates'][0]['critic_other_parameter_delta_l2_from_bc']:.6f}. "
    "Optimization is verified, but PPO did not improve final success over BC (both82/100). "
    "A continuation to34,816 total transitions regressed to2/20 validation success and was not selected. "
    "Teacher reference counterfactuals on10 additional test scenes ×4 instructions yielded40/40 "
    "for the script and34/40 for selected PPO, with identical initial geometry across instructions.",
    f"\nA separate, non-selected 1,024-transition CPU profiling continuation ran at "
    f"{teacher_resource['transitions_per_second']:.1f} transitions/s, "
    f"RSS {gib(teacher_resource['peak_process_rss_bytes'])} GiB and private committed memory "
    f"{gib(teacher_resource['peak_process_private_bytes'])} GiB. These auxiliary updates did not produce benchmark data or selected weights.",
    "\nThe raw collection contains274 lossless HDF5 episodes:250 train (208 success,42 failure) "
    "and24 validation (21 success,3 failure), approximately2.09GB. The first200 successful train "
    "episodes and all21 successful validation episodes are identically selected for every student. "
    "Per-file SHA256, teacher provenance, scene seeds, images/depth/calibration, timestamps, states "
    "and actions were audited. No test scene occurs in training, including teacher training. "
    "Failures are retained but are not used for imitation optimization.",
    f"\nNative LeRobot export was actually reloaded: {native_export['episodes']} episodes / "
    f"{native_export['frames']:,} frames. RGB PNG, state/actions Parquet and actual task text are "
    "provided, with indexed HDF5 sidecars for metric depth, point clouds, extrinsics, timestamps "
    "and source hashes. Reloaded source RGB/state/actions match exactly.",
    "\n## Principal closed-loop results",
    "\n"
    + table(
        [
            "Condition",
            "Mono seed0/1/2",
            "Fusion seed0/1/2",
            "ACT seed0/1/2",
            "n per seed mono/fusion; ACT",
        ],
        [
            [condition_names[c]]
            + [
                " / ".join(percent(v) for v in aggregate[m, c]["seed_rates"])
                for m in ["mono", "fusion", "act"]
            ]
            + [f"{100 if c in ['nominal','occlusion','fixed_camera_missing'] else 20}; 20"]
            for c in conditions
        ],
    ),
    "\n![Success rates](../results/figures/success_rates.png)",
    "\n"
    + table(
        [
            "Condition",
            "Fusion mean ± seed SD",
            "Mono mean ± seed SD",
            "Paired fusion−mono (pp)",
            "Crossed bootstrap95% (pp)",
        ],
        [
            [
                condition_names[c],
                f"{100*aggregate['fusion',c]['mean_success_rate']:.1f} ± {100*aggregate['fusion',c]['seed_std']:.1f}%",
                f"{100*aggregate['mono',c]['mean_success_rate']:.1f} ± {100*aggregate['mono',c]['seed_std']:.1f}%",
                f"{comparison[c]['fusion_minus_mono_pp']:+.1f}",
                "["
                + ", ".join(f"{v:.1f}" for v in comparison[c]["paired_crossed_bootstrap95_pp"])
                + "]",
            ]
            for c in conditions
        ],
    ),
    "\n![Paired differences](../results/figures/paired_fusion_difference.png)",
    "\nThe bootstrap resamples training seeds and shared paired scene columns together (5,000 draws). "
    "Individual seed Wilson intervals and all raw metrics are in the JSON/CSV outputs. Three seeds "
    "give limited training-variability estimation; frames are not independent observations. "
    "Depth/extrinsic cells and ACT have20 scenes per seed and remain exploratory. The OOD drop "
    "is paired with the same nominal scene subset, including first20 rather than all100 when appropriate. "
    "RGB-only ACT sees no change under depth-only or extrinsic-only perturbation; those conditions are input no-ops.",
    "\n"
    + table(
        ["Mode / condition", "Paired nominal−OOD drop, seed0/1/2 (pp)"],
        [
            [
                m + " / " + condition_names[c],
                " / ".join(
                    f"{next(r['nominal_minus_ood_pp'] for r in primary['ood_drop'] if r['mode']==m and r['condition']==c and r['training_seed']==s):+.1f}"
                    for s in [0, 1, 2]
                ),
            ]
            for m in ["mono", "fusion", "act"]
            for c in conditions[1:]
        ],
    ),
    "\n"
    + table(
        [
            "Mode",
            "Nominal wrong target",
            "Wrong destination",
            "Drop",
            "Collision",
            "Mean terminal XY error (cm)",
        ],
        [
            [m]
            + [
                percent(
                    np.mean(
                        [
                            r[key]
                            for r in primary["per_seed"]
                            if r["mode"] == m and r["condition"] == "nominal"
                        ]
                    )
                )
                for key in ["wrong_target", "wrong_destination", "dropped", "collision"]
            ]
            + [
                f"{100*np.mean([r['placement_error_m'] for r in primary['per_seed'] if r['mode']==m and r['condition']=='nominal']):.2f}"
            ]
            for m in ["mono", "fusion", "act"]
        ],
    ),
    "\nTerminal placement errors include failed rollouts. Wrong-target flags include any non-target "
    "cube lift/tray entry and may coexist with eventual target success. Rates therefore are not "
    "exclusive categories.",
    "\n## Instruction counterfactuals and ablations",
    "\nEach student counterfactual cell reuses10 physical scenes with all4 instructions. "
    "Hash equality of initial RGB, depth, allowed robot state and camera poses is asserted across "
    "instructions. Changes in first action indicate conditioning, while all-four completion is the stricter measure.",
    "\n"
    + table(
        [
            "Mode / seed",
            "Success /40",
            "Scenes with all4 successes /10",
            "Object / goal first-action ΔL2",
        ],
        [
            [
                f"{r['mode']} / {r['training_seed']}",
                r["success_count"],
                r["scenes_all_four_success"],
                f"{r['mean_first_action_l2_change_object_instruction']:.3f} / {r['mean_first_action_l2_change_goal_instruction']:.3f}",
            ]
            for r in secondary["counterfactuals"]
        ],
    ),
    "\n"
    + table(
        ["Variant / condition", "Successful /20", "Baseline /20", "Paired difference (pp)"],
        [
            [
                r["variant"] + " / " + r["condition"],
                r["success_count"],
                r["baseline200_success_count"],
                f"{r['paired_success_difference_pp']:+.1f}",
            ]
            for r in secondary["single_seed_ablations"]
        ],
    ),
    "\n![Secondary ablations](../results/figures/secondary_ablations.png)",
    "\nThese are one-seed,20-paired-scene checks. The50-demo variant uses the first50 of the "
    "same successful demonstrations at equal8,000-update budget; this is not a data-efficiency "
    "curve. View dropout retains one random view with20% probability and uses a separate "
    "checkpointed RNG to preserve minibatch frame sampling. No-voxel changes inference filtering "
    "only with identical512-point budget and fixed weights; it is not a retrained ablation. "
    "A factorial study of filtering/pretraining/architecture is outside this experiment.",
    "\n## SmolVLA and negative experiments",
    f"\nAuthentic SmolVLA received500 optimizer updates (batch1, accumulation4, one seed) on "
    f"the same200 demonstrations in {vla_train['wall_seconds']:.1f}s optimization-window time. "
    "The frozen pretrained VLM and official preprocessing were retained. Trainable modules "
    "are natively BF16/FP32; no additional AMP/scaler was enabled. A real forward/backward/optimizer "
    "pilot passed before fine-tuning. Pinned upstream model/tokenizer revisions and adapter provenance are retained.",
    f"\nFinal SmolVLA nominal success: **{secondary['smolvla']['success_count']}/20**; "
    f"wrong target {secondary['smolvla']['wrong_target_count']}/20, collisions {secondary['smolvla']['collision_count']}/20. "
    f"Mean of per-episode policy P50/P95: {secondary['smolvla']['mean_episode_policy_p50_ms']:.1f} / "
    f"{secondary['smolvla']['mean_episode_policy_p95_ms']:.1f}ms per8-action chunk. "
    "Execution uses only2 actions per inference; this does not establish20Hz real-time deployment.",
    (
        "\nSmolVLA OOD was deferred because0/20 nominal success triggered the predeclared secondary-budget gate; "
        "fine-tuning itself succeeded, and deferral is not attributed to OOM."
        if secondary["smolvla"]["ood_deferred_if_zero_nominal"]
        else "\nSmolVLA OOD successful /20: "
        + "; ".join(f"{c}: {n}" for c, n in secondary["smolvla"]["ood"].items())
        + "."
    ),
    "\nRetained negative pilots: epsilon-prediction diffusion40d/1,000updates yielded0/20 "
    "validation successes despite decreasing loss; sample-prediction/legacy-prior200d/2,000updates "
    "yielded2/20 (8/20 lifted). Final chroma40 recipe was chosen using validation before freezing "
    "the main protocol. A heuristic observation-aliasing diagnostic on200 train episodes found "
    "zero candidates under its near-static-state/label-jump criteria. It does not demonstrate "
    "aliasing; limitations of current-frame policies and timed teacher phases remain hypotheses.",
    "\n## Runtime, resources and deployment export",
    "\nPolicy-only profiles warm up10 calls and time50 batch1 chunks, excluding rendering and "
    "preprocessing. Compact weights are FP32; native CUDA kernel dispatch may use default TF32. "
    "The main benchmark uses two-thread CPU PyTorch consistently. GPU microprofiles show small "
    "compact models can be slower on this workload; dispatch overhead is a hypothesis, not a traced conclusion.",
]

profiles = {m: read(f"results/profile_{m}.json") for m in ["mono", "fusion", "act"]}
latency_rows = []
for mode, profile in profiles.items():
    for measurement in profile["measurements"]:
        latency_rows.append(
            [
                mode,
                measurement["device"],
                measurement["threads"],
                f"{measurement['p50_ms']:.2f}",
                f"{measurement['p95_ms']:.2f}",
            ]
        )
lines += [
    "\n" + table(["Policy", "Device", "CPU threads", "Policy P50 (ms)", "P95 (ms)"], latency_rows)
]
lines += [
    "\n"
    + table(
        [
            "Mode",
            "Mean episode preprocess P50/P95 (ms)",
            "Policy P50/P95 (ms)",
            "Total P50/P95 (ms)",
        ],
        [
            [m]
            + [
                " / ".join(
                    f"{np.mean([r[prefix+'_'+q+'_ms'] for r in primary['per_seed'] if r['mode']==m and r['condition']=='nominal']):.2f}"
                    for q in ["p50", "p95"]
                )
                for prefix in ["preprocess", "policy", "total"]
            ]
            for m in ["mono", "fusion", "act"]
        ],
    )
]
lines += [
    "\nOnline latency values are means of episode percentiles, not pooled-call percentiles. "
    "Total is preprocessing+policy and excludes physics/control/render; it is not end-to-end wall time. "
    "Raw rollouts also record wall duration and simulated duration.",
    "\n"
    + table(
        ["Mode", "Mean nominal wall duration (s)", "Mean simulated duration (s)"],
        [
            [m]
            + [
                f"{np.mean([r[key] for r in primary['per_seed'] if r['mode']==m and r['condition']=='nominal']):.2f}"
                for key in ["wall_seconds", "simulation_seconds"]
            ]
            for m in ["mono", "fusion", "act"]
        ],
    ),
    "\n"
    + table(
        [
            "Training run",
            "Selected EMA update /8,000",
            "Optimization/validation/I/O seconds",
            "Peak Torch allocated GiB",
        ],
        [
            [
                r["run"],
                selected_updates[r["run"]],
                f"{r['wall_seconds']:.1f}",
                gib(r["peak_torch_vram_bytes"]),
            ]
            for r in training
        ],
    ),
    "\nTraining manifest times exclude dataset loading/initialization. Matrix process times include "
    "those costs. Torch allocator peaks exclude other GPU processes, driver and renderer allocations.",
    "\nActual combined gates kept both RGB-D cameras alive during a trained model optimizer step "
    "with loaded AdamW moments; fresh rendering then ran while optimizer state remained allocated. "
    "Disposable copies were used and selected checkpoints were unchanged. GPU figures below are "
    "whole-device snapshots, not per-process attribution; host private memory differs from RSS.",
    "\n"
    + table(
        [
            "Gate",
            "Batch",
            "Torch allocated/reserved GiB",
            "GPU used/free MiB; °C",
            "RSS/private GiB",
            "Probe ΔL2",
        ],
        [
            [
                r["mode"],
                r["batch_size"],
                f"{gib(r['peak_torch_allocated_bytes'])} / {gib(r['peak_torch_reserved_bytes'])}",
                r["gpu_used_free_mib_temperature_c"],
                f"{gib(r['process_rss_bytes'])} / {gib(r['process_private_bytes'])}",
                f"{r['probe_delta_l2']:.6g}",
            ]
            for r in compact_resources + vla_resources
        ],
    ),
    f"\nONNX opset17 exports only the FP32 temporal denoiser. Point encoding, preprocessing and "
    f"the10-step DDIM loop remain PyTorch. Maximum subgraph error is {onnx['max_abs_error']:.3g}; "
    f"full normalized DDIM action error {onnx['full_ddim_normalized_action_max_abs_error']:.3g}. "
    f"CPU2 policy-only medians are {onnx['full_policy_cpu2_measurements']['pytorch']['p50_ms']:.2f}ms "
    f"PyTorch and {onnx['full_policy_cpu2_measurements']['onnx_denoiser']['p50_ms']:.2f}ms with ONNX denoiser. "
    "The separate20-scene paired validation gives6/20 for both,20/20 success labels agree, "
    "18/20 collision labels agree and terminal placement differs by up to1.34mm. Final paired "
    "test results appear in the ablation table. Physics equivalence is not bit-exact. TensorRT/Jetson were not executed.",
    "\nA future RGB-D/action interface validates shapes, units, transforms and command limits; "
    "the calibration procedure is documented. No physical robot transfer, hardware connection "
    "or Isaac Sim/Lab run occurred.",
    "\n## Demonstrations and reproduction",
    "\nThe success and failure videos replay the earliest successful and failed fusion seed0 "
    "nominal test episodes respectively (200001/200000). Replay actions, steps and final placement "
    "match the source metrics exactly. Videos show actual fixed/wrist simulator frames, "
    "including the final state; magnification uses nearest-neighbor sampling.",
    "\n![Success replay](../results/demos/success.gif)\n\n![Failure replay](../results/demos/failure.gif)",
    "\nA fresh `.venv-repro` was installed from portable pinned dependencies, passed dependency "
    "consistency checks and the CPU suite, then rendered both Lift RGB-D views and a full episode. "
    "Its selected checkpoint replay reproduced a validation success with identical first action, "
    "steps and terminal placement (zero difference). The opt-in CUDA test separately verified "
    "an exactly identical next optimizer update after restoring model/optimizer/scheduler and RNG states. "
    "An actual full-training continuation from mono seed1 update7,500 to8,000 reproduced all500 "
    "updates: raw weights, EMA, AdamW moments, scheduler, normalization, config, progress and "
    "Python/NumPy/Torch/CUDA plus minibatch RNG states are bit-identical to the original run. "
    "Only a disposable copy was optimized; selected benchmark weights were preserved. "
    "Remote GitHub CI was not executed. Full benchmark rerunning still needs the local data and model artifacts.",
    "\n## Limits, artifacts and practiced skills",
    "\nThe benchmark tests a narrow simulated scene distribution and four instruction meanings. "
    "Calibration is known, sensor corruption is synthetic, success is instantaneous, and only "
    "three training seeds are used. Low success rates and wrong-target behavior must be read "
    "alongside any fusion advantage. BC supplies most teacher competence; no PPO improvement "
    "is claimed. SmolVLA/pretraining/ACT results cannot isolate architecture effects at equal compute. "
    "Secondary20-scene checks do not support a broad data-efficiency or robustness claim.",
    "\nExecuted skills: simulated control and reward design; supervised actor initialization and "
    "on-policy PPO updates; lossless provenance-aware robot trajectory collection; dynamic RGB-D "
    "geometry/fusion; ACT/CVAE and diffusion policy training; authentic VLA fine-tuning; paired "
    "multi-seed closed-loop evaluation; resource profiling; partial ONNX export; exact checkpoint "
    "resume and clean-environment reproduction. These do not imply real-robot, Isaac or Jetson experience.",
    "\nTracked artifacts: [principal raw CSV](../results/raw/student_rollouts.csv), "
    "[per-seed CSV](../results/per_seed.csv), [principal summary](../results/benchmark_summary.json), "
    "[secondary summary](../results/secondary_summary.json), figures and lightweight demos. "
    "Full local weights/data remain ignored under `artifacts/`; see [publication instructions](artifact_publication.md). "
    "Protocol and dataset/run hashes are in `configs/` and `results/checkpoint_audit.json`.",
    "\n## Proposed CV bullet — executed facts only",
    "\n« Développement d’un benchmark local de robot learning sous MuJoCo/PyTorch : collecte "
    "traçable de274 trajectoires Panda issues d’un acteur BC+PPO, entraînement ACT et diffusion "
    "3D mono/deux vues sur3 seeds, évaluation en2 340 rollouts, fine-tuning SmolVLA et validation "
    "d’un export ONNX partiel sur RTX4060 8Go. »",
    "\nThis is a proposal only. The factual CV/profile files were not modified.",
]

fig = Path("results/figures")
fig.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
f, ax = plt.subplots(figsize=(7, 4))
names = list(teacher)
rates = [sum(r["success"] for r in teacher[n]) for n in names]
ax.bar(names, rates, color=["#8c929b", "#48a28a", "#4567ae", "#e18432"])
for i, v in enumerate(rates):
    ax.text(i, v + 2, f"{v}/100", ha="center")
ax.set_ylim(0, 111)
ax.set_ylabel("Instruction-correct success (%)")
ax.set_title("Final teacher controls — 100 paired test scenes")
f.tight_layout()
for ext in ["png", "svg"]:
    f.savefig(fig / f"teacher_final_controls.{ext}", dpi=180)
plt.close(f)
lines.insert(
    lines.index("\n## Principal closed-loop results"),
    "\n![Final teacher controls](../results/figures/teacher_final_controls.png)",
)
report_text = "\n".join(lines) + "\n"
report_text = re.sub(
    r"\b(all|same|the|with|for|on|and|over|under|first|by|at|every|only|to|contains|uses|received|completed|yielded|achieved|failed|seed|batch|accumulation|de|sur|en)(?=\d)",
    r"\1 ",
    report_text,
)
report_text = re.sub(
    r"(?<=\d)(?=updates\b|transitions\b|episodes\b|scenes\b|frames\b|same\b|paired\b|ms\b|cm\b|mm\b|GB\b|GiB\b|Go\b|Hz\b)",
    " ",
    report_text,
)
report_text = report_text.replace("RTX4060", "RTX 4060").replace("500M vs", "500 M vs")
Path("docs/final_report.md").write_text(report_text, encoding="utf8")
print("Measured report written: docs/final_report.md")
