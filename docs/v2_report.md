# V2 measured before/after report

All results below were actually executed locally. V1 is retained at tag `geopolicy-v1-2026-09-30` (542f132); its original results, weights, source and historical report are unchanged. The new evaluation uses a stricter stable-placement criterion and longer 240-step horizon, so historical instantaneous rates are contextual figures. The before/after table re-evaluates preserved V1 checkpoints on exactly the same new scenes and metric.

**Outcome:** the checks and reproduction are stronger, but no nominal manipulation improvement over preserved V1 is demonstrated. Observed fusion successes were V1 16/150 and V2 10/150; ACT was V1 9/150 and V2 0/150. The targeted ACT search did not produce a reliable learned baseline. The fusion advantage over wrist-only and the nominal manual-prior effect remain uncertain; the full paired intervals and negative experiments are below. V1 remains the performance reference, and V2 is a separate diagnostic study.

## Question and controls

Does calibrated fixed+wrist RGB-D fusion help beyond either view alone, and how much does the manual color prior contribute? Six diffusion variants cross fixed/wrist/fusion with prior present/absent. All use the same 200 training episodes extended with 30 actual PPO-teacher steps after the original end, the same 21 validation episodes, frozen V1 normalization, 7D OSC actions, 512 points, 8-step prediction, 2-step execution, 8,000 updates, batch 32 and three training seeds. The prior ablation removes the additive chromatic attention bias; it retains learned attention and explicit XYZRGB moments. RGB still contains colors. The models are compact independent adaptations, without pretrained vision.

The selected ACT uses `act_history_binary`: a compact random-CNN RGB action-chunk Transformer, trained directly on its zero-latent deployment path, with the selected proprioceptive history and gripper output. Its modality and capacity differ from diffusion; its equal optimizer budget does not establish architectural parity with original ACT. The rejected `act_point` was an explicit 3D adaptation, not an official RGB ACT reproduction. PPO and SmolVLA remain the unchanged V1 extensions; no new improvement is claimed for them.

## Diagnoses before increasing budgets

Recorded versus live first-frame ACT RGB and normalized state match exactly on scene 100000. Action normalization roundtrip max error is 5.96e-08. The physical Panda probe verified negative command opens and positive command closes; the backend uses command sign, so a small positive value is not a weak close. The new binary output maps to physical ±1 after normalization, tested with a nonzero action mean. Live history copies robot state arrays at observation time to avoid MuJoCo buffer aliasing and uses no future state or privileged object pose.

V1 ACT on recorded validation: normalized L1 prior 0.054961, future-action-conditioned posterior oracle 0.054960; gripper sign accuracy 99.83%. The tiny posterior/prior difference does not demonstrate a latent train/inference mismatch causing the failures. Zero-latent inference is also used in [original ACT](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py) and [LeRobot ACT 0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/modeling_act.py). Correct per-frame preprocessing and low offline error do not establish closed-loop competence.

The first-frame input sensitivity interventions are in `results/v2/act_input_sensitivity.json`. Image permutation and instruction flips change the output locally, but this does not prove grounded task behavior or identify a unique cause. Grasp and selection dominate the fresh V1 validation failures. V1 ACT's historical 0/60 nominal test remains unchanged; it achieved 2/20 on the new stable-metric diagnostic validation set, so '0%' is not an intrinsic impossibility claim. The scripted diagnostic reference achieved 20/20 stable placements; students receive none of its truth-pose waypoints.

| Validation ACT control | Stable /20 | Meaning |
| --- | --- | --- |
| Unassisted preserved ACT | 2 | Learned controller |
| Scripted privileged motion + ACT gripper | 16 | Diagnostic intervention only |
| ACT motion + scripted privileged gripper | 2 | Diagnostic intervention only |

These paired component interventions use the same 100200–100219 validation scenes. Replacing the six motion commands raises the score from 2/20 to 16/20; replacing only the gripper gives 2/20. This localizes a substantial bottleneck to the learned motion-command path under these interventions, rather than establishing a gripper-sign bug. The script uses ground-truth object/goal waypoints and privileged phase timing, so the intervention does not isolate perception, temporal prediction or architectural capacity as a unique cause. No oracle output counts as learned-baseline improvement, no final-test oracle is allowed, and these diagnostic results do not retune the frozen main choices. The unassisted score and stage traces remain the baseline evidence.

The original demonstrations terminate at the instantaneous metric. All 221 exact-prefix trajectories were continued for 30 actual selected-PPO-teacher steps; all extended teacher trajectories achieved sustained stability. This demonstrates that recording covers post-release behavior. In a matched 3,000-update seed-0 control, original data gave 4/20 stable placements versus 1/20 after continuation. This result does not support continuation as a performance improvement. The original-data control was outside the preset ranking: the final view/prior study intentionally uses one shared extended dataset to cover the new scoring interval, rather than claiming it was the best dataset on validation. That design choice and possible degradation from the added data are limitations; the full V2 before/after comparison must establish the resulting behavior. This small single-seed control neither isolates every mechanism nor demonstrates a general gain. All negative pilots and source data are retained.

## Intermediate implementation correction

An intermediate V2 binary-gripper variant left the seventh diffusion output unsupervised while still using it as part of the iterative DDIM state. On the same diagnostic batch its seventh-output gradient L1 was 0 without auxiliary supervision and 7.00354 with it. This is a demonstrated structural defect introduced in V2's intermediate adaptation, not in preserved V1. Its contribution to manipulation failures was not isolated causally. A 0.1-weight normalized-gripper MSE now trains that internal channel while the separate classifier still provides the physical sign command. Seven targeted tests cover this, rotated-corner containment and the existing contracts.

Before any reserved test, ten completed intermediate 8,000-update runs and one partial checkpoint, plus all pilots/initial confirmation, were archived separately in `artifacts/v2/intermediate_v2a`. Their negative results are retained in `results/v2/intermediate_v2a_summary.json` and the validation raw table. Pilot selection was repeated after the correction using the validation set; final independent confirmation uses 100280–100299 for diffusion and 100320–100339 for ACT. Final budgets remain 8,000 updates per model. The additional compute is recorded as intermediate diagnostic work, not silently counted as one training run or used to select on the test.

## Validation-only selection

Six initial corrected 3,000-update pilots used the same optimizer and data budget. A seventh RGB ACT pilot adaptively broadcasts the learned instruction embedding to all visual tokens, motivated by the recorded first-frame sensitivity probe. It adds no parameters and keeps the same data/update budget; its 0/20 stable result rejects this improvement on validation. The original ranking rule remained unchanged. Selection rule was recorded before comparative results: most stable successes, then fewer wrong-object lifts, fewer collisions, more correct transport, then declared order. Selected diffusion: `diffusion_history`; selected ACT: `act_history_binary`. Disjoint confirmation: diffusion 100280–100299, ACT 100320–100339, reported without retuning: act: 0/20, diffusion: 0/20.

| Pilot | Stable /20 | Wrong lift | Collision episodes | Correct transport |
| --- | --- | --- | --- | --- |
| diffusion_suffix_only | 1 | 3 | 5 | 1 |
| diffusion_binary | 2 | 3 | 8 | 2 |
| diffusion_history | 2 | 2 | 1 | 2 |
| act_prior_only | 1 | 4 | 8 | 1 |
| act_history_binary | 1 | 3 | 11 | 1 |
| act_point | 0 | 2 | 0 | 0 |
| act_instruction_broadcast | 0 | 0 | 20 | 0 |

The additional ACT diagnostic is specified in `configs/v2/act_diagnostic.json`; earlier ACT confirmation is retained separately. Main choices, hashes, budgets and source identities were frozen in `configs/v2/final_protocol.json` before any new reserved test. Final scenes 300000–300049 are distinct from the V1 test and all validation scenes. Results cannot change the selected recipe. Three training seeds are crossed with 50 identical scenes per condition. Bootstrap resamples training seeds and paired scene columns jointly (10,000 draws, RNG 81); percentile intervals are descriptive, with only three training seeds, no multiplicity correction, and no claim about unseen scene families. An all-zero observed sample does not establish a zero population probability.

## New reserved test

| Model | Condition | Stable /150 | Seeds (success /50) | Ever V1 instant /150 |
| --- | --- | --- | --- | --- |
| fixed_prior | nominal | 10/150 (6.7%) | [5, 3, 2] | 11 |
| fixed_prior | occlusion | 9/150 (6.0%) | [4, 1, 4] | 9 |
| fixed_prior | fixed_camera_missing | 1/150 (0.7%) | [0, 1, 0] | 1 |
| fixed_no_prior | nominal | 8/150 (5.3%) | [4, 2, 2] | 8 |
| fixed_no_prior | occlusion | 8/150 (5.3%) | [4, 2, 2] | 8 |
| fixed_no_prior | fixed_camera_missing | 0/150 (0.0%) | [0, 0, 0] | 0 |
| wrist_prior | nominal | 7/150 (4.7%) | [2, 4, 1] | 8 |
| wrist_prior | occlusion | 7/150 (4.7%) | [2, 4, 1] | 8 |
| wrist_prior | fixed_camera_missing | 7/150 (4.7%) | [2, 4, 1] | 8 |
| wrist_no_prior | nominal | 6/150 (4.0%) | [3, 1, 2] | 7 |
| wrist_no_prior | occlusion | 6/150 (4.0%) | [3, 1, 2] | 7 |
| wrist_no_prior | fixed_camera_missing | 6/150 (4.0%) | [3, 1, 2] | 7 |
| fusion_prior | nominal | 10/150 (6.7%) | [3, 4, 3] | 10 |
| fusion_prior | occlusion | 12/150 (8.0%) | [5, 4, 3] | 12 |
| fusion_prior | fixed_camera_missing | 14/150 (9.3%) | [3, 7, 4] | 14 |
| fusion_no_prior | nominal | 5/150 (3.3%) | [2, 3, 0] | 8 |
| fusion_no_prior | occlusion | 9/150 (6.0%) | [4, 4, 1] | 10 |
| fusion_no_prior | fixed_camera_missing | 3/150 (2.0%) | [1, 2, 0] | 4 |
| act_selected | nominal | 0/150 (0.0%) | [0, 0, 0] | 0 |
| act_selected | occlusion | 0/150 (0.0%) | [0, 0, 0] | 1 |
| act_selected | fixed_camera_missing | 1/150 (0.7%) | [0, 1, 0] | 1 |

![Three-seed stable placement](../results/v2/figures/main_comparison.png)

| Paired comparison | Condition | Difference pp | Descriptive 95% interval |
| --- | --- | --- | --- |
| fusion_prior − fixed_prior | nominal | +0.0 | [-5.3, +5.3] |
| fusion_prior − wrist_prior | nominal | +2.0 | [-3.3, +7.3] |
| fixed_prior − fixed_no_prior | nominal | +1.3 | [-4.7, +8.0] |
| wrist_prior − wrist_no_prior | nominal | +0.7 | [-6.0, +8.0] |
| fusion_prior − fusion_no_prior | nominal | +3.3 | [-2.7, +9.3] |
| fusion_prior − fixed_prior | occlusion | +2.0 | [-5.3, +8.7] |
| fusion_prior − wrist_prior | occlusion | +3.3 | [-4.0, +10.0] |
| fixed_prior − fixed_no_prior | occlusion | +0.7 | [-6.0, +7.3] |
| wrist_prior − wrist_no_prior | occlusion | +0.7 | [-6.0, +8.0] |
| fusion_prior − fusion_no_prior | occlusion | +2.0 | [-4.7, +8.7] |
| fusion_prior − fixed_prior | fixed_camera_missing | +8.7 | [+2.7, +16.0] |
| fusion_prior − wrist_prior | fixed_camera_missing | +4.7 | [-2.0, +12.0] |
| fixed_prior − fixed_no_prior | fixed_camera_missing | +0.7 | [+0.0, +4.0] |
| wrist_prior − wrist_no_prior | fixed_camera_missing | +0.7 | [-6.0, +8.0] |
| fusion_prior − fusion_no_prior | fixed_camera_missing | +7.3 | [+0.0, +15.3] |

The protocol marks four primary comparisons; other view/prior differences are exploratory. A positive point estimate alone is not proof of an advantage. The wrist-only camera-loss cell is a control for the surviving sensor; it helps distinguish fusion from merely retaining that view. Manual prior effects are separated within each view under an otherwise common recipe.

## Fair before/after

| Model | V1 stable /150 | V2 stable /150 | Paired difference pp [95%] |
| --- | --- | --- | --- |
| fusion | 16 | 10 | -4.0 [-13.3, +4.0] |
| act | 9 | 0 | -6.0 [-14.7, +0.0] |

The full V2 recipe bundles continued data, causal history, loss/output changes and selected training paths. These comparisons demonstrate the resulting behavior in this scene family; they do not attribute a gain to one component. V1's historical nominal fusion 30.7%, fixed 30.3%, missing-fixed fusion 17.0% versus fixed 1.0% used instantaneous success and a different test set. They remain documented in the untouched `docs/final_report.md` and cannot be substituted for this stable-score comparison.

## Failure localization and instruction controls

| Model | Condition | Selection/no approach | Grasp | Transport | Release | Post-release criterion |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_prior | nominal | 68 | 52 | 19 | 1 | 0 |
| fixed_prior | occlusion | 72 | 50 | 18 | 1 | 0 |
| fixed_prior | fixed_camera_missing | 113 | 23 | 12 | 1 | 0 |
| fixed_no_prior | nominal | 72 | 58 | 12 | 0 | 0 |
| fixed_no_prior | occlusion | 71 | 56 | 15 | 0 | 0 |
| fixed_no_prior | fixed_camera_missing | 134 | 10 | 6 | 0 | 0 |
| wrist_prior | nominal | 73 | 47 | 22 | 1 | 0 |
| wrist_prior | occlusion | 73 | 47 | 22 | 1 | 0 |
| wrist_prior | fixed_camera_missing | 73 | 47 | 22 | 1 | 0 |
| wrist_no_prior | nominal | 76 | 52 | 15 | 1 | 0 |
| wrist_no_prior | occlusion | 76 | 52 | 15 | 1 | 0 |
| wrist_no_prior | fixed_camera_missing | 76 | 52 | 15 | 1 | 0 |
| fusion_prior | nominal | 72 | 46 | 22 | 0 | 0 |
| fusion_prior | occlusion | 74 | 44 | 20 | 0 | 0 |
| fusion_prior | fixed_camera_missing | 78 | 43 | 15 | 0 | 0 |
| fusion_no_prior | nominal | 67 | 61 | 14 | 1 | 2 |
| fusion_no_prior | occlusion | 70 | 58 | 12 | 1 | 0 |
| fusion_no_prior | fixed_camera_missing | 68 | 60 | 18 | 1 | 0 |
| act_selected | nominal | 100 | 48 | 2 | 0 | 0 |
| act_selected | occlusion | 97 | 50 | 2 | 1 | 0 |
| act_selected | fixed_camera_missing | 96 | 52 | 1 | 0 | 0 |

Stages are evaluator-only geometric/contact proxies: selection (wrong object lifted without selected lift, or no target approach), grasp (no selected lift), transport (no correct destination reach), release (no full contained open-finger release), unstable (release seen but no continuous stable interval). They are mutually exclusive failure labels, not a claim that the teacher phase machine was inferred. The JSON label `unstable_placement` means the complete post-release criterion was not held for one second: it can reflect hand closure even when the cube itself remains still. It is not proof of physical object instability; positions, speeds and contacts in the trace distinguish these cases. Collision and wrong-target flags remain separate; success is not certified collision-free.

| Model | All four /30 scene-seed pairs | Per training seed | Individual /120 |
| --- | --- | --- | --- |
| fusion_prior | 0 | [0, 0, 0] | 21 |
| fusion_no_prior | 0 | [0, 0, 0] | 21 |
| act_selected | 0 | [0, 0, 0] | 0 |

All four instructions reuse identical RGB, depth, camera poses and robot state at initialization; sensor hashes were verified across views/seeds and instruction changes. Instruction tokens change; no object truth enters student inputs. The four-value instruction vector concatenates two one-hot pairs (red/green object and blue/yellow destination); the displayed sentence is generated from these labels, with no learned text encoder. Strict all-four completion and individual rates constrain grounding claims across these four known combinations; this is not open-vocabulary language understanding.

## Stable placement contract

At 20 Hz, valid samples must span one continuous second (at least 21 samples). The selected cube must previously lift above 0.89 m, have center height 0.816–0.855 m and all rotated XY corners contained in the chosen tray's inner half-extents [0.063, 0.060] m minus 1 mm margin. Fingers must be open at least 45 mm with no selected-object finger contact, object linear speed ≤0.02 m/s and angular speed ≤0.25 rad/s. Invalid samples reset the interval. This primary protocol also requires fingers to remain open throughout the dwell; closing empty fingers after withdrawing can therefore fail the strict score despite an otherwise stable object. This conservative robot-ending-posture requirement is explicit, and raw traces retain object pose/speed and hand width so it is not misattributed to object motion. The policy keeps acting throughout verification, with no assisted release. Evaluation ends at stable success or 240 steps. The tray dimensions are specific to this simulator asset; the criterion measures a one-second dwell, not indefinite stability or general collision safety. Unit tests cover interruption, transient crossing, contact, finger width, speed, containment and prior lift.

## Execution, verification and reproduction

One GPU job at a time on RTX 4060 Laptop 8 GB; all optimizations FP32. Per-preset actual AdamW/dual-render memory gates and exact next-update recovery passed. Main training resource summary: `{"maximum_gpu_used_mib": 814.0, "minimum_free_gpu_mib": 7143.0, "maximum_temperature_c": 49.0, "maximum_process_private_gib": 3.296672821044922, "minimum_free_commit_gib": 6.306247711181641, "minimum_free_disk_gib": 42.47411346435547, "summed_main_optimization_seconds": 3939.7819999999965, "maximum_torch_allocated_mib": 142.38134765625}`. The field `optimization_seconds` is the elapsed training-loop clock, including periodic validation and checkpoint writes; its summed main value covers the 21 final runs, not all archived diagnostic compute. CUDA allocation excludes driver/context/renderer memory; device-wide and process committed memory are logged separately. Training gates monitor temperature, disk, GPU and Windows commit headroom, preserve periodic checkpoints, and stop on persistent unsafe samples. Main and evaluation manifests contain timings and resource traces; wall time includes local hardware effects.

Executed final counts: 3150 main, 300 fair before/after, 360 counterfactual rollouts. Raw CSV/JSON live in `results/v2/raw`; per-step traces and checkpoints remain in local ignored `artifacts/v2`. Before opening final test, all original 274 HDF5 hashes, 42 original checkpoints and 122 tracked V1 files passed checks; all 221 augmented prefixes are array-exact. Continuing the real fusion checkpoint for its last 500 updates reproduced model, optimizer, scheduler, EMA and configuration exactly. Test outputs are hash-bound to frozen checkpoints and source. See `results/v2/pretest_verification.json` and `results/v2/v1_preservation.json`.

The compact local demo bundle contains only an EMA policy, recipe, hashes and raw expected examples. It needs no training data or pretrained-model download. The earliest success and earliest failure of predeclared fusion-prior seed 0 are shown, when present; this selection is disclosed and is not aggregate evidence. A separate pinned `.venv-repro` passed V2 tests and replayed the compact checkpoint with every action and selected-object position identical, plus matching steps, initial sensor hashes and success labels. A second replay used an isolated copy of the source and bundle with no training dataset or original training weights present. See [clean replay](../results/v2/clean_reproduction.json), [isolated replay](../results/v2/portable_demo_verification.json) and [reproduction instructions](v2_reproduction.md). The final [delivery audit](../results/v2/delivery_verification.json) recomputes the strict dwell from all recorded final traces and verifies frozen identities and ZIP contents; XY containment uses the saved evaluator flag rather than an independent geometry implementation.

## Limits and defensible claims

Absolute manipulation reliability remains the limiting factor. Three seeds and 50 scenes cannot establish broad robot-learning superiority. Simulation uses known calibration, rigid colored cubes, two one-hot instruction pairs, selected successful demonstrations and a privileged phase-based BC-initialized PPO teacher. No real robot, Isaac, Jetson, open-vocabulary instruction or sim-to-real transfer was executed. Hardware timings apply to this PC. Negative ACT/3D pilots and failed V1 PPO/SmolVLA runs remain visible. The scientific V2 study was completed locally before the later GitHub presentation request. The CV was not modified.
