# GitHub presentation — 2026-10-01

Status: COMPLETE — public repository, release, GitHub CI and fresh-clone replay verified. Repository: https://github.com/Alecbossard/GeoPolicy-Bench. Release: https://github.com/Alecbossard/GeoPolicy-Bench/releases/tag/v2-demo. The user requested GitHub publication after the scientific V2 snapshot below. No new training or changes to frozen scientific outcomes are part of this presentation work.

The curated front page includes actual success/failure animations, balanced results, reproduction and limits. A compact release bundle with SHA-256 verification supports the short demo. CPU contracts and publication/link checks run in GitHub Actions; full data and training artifacts remain local. GitHub CI passed 19 CPU tests and 78 publication/link checks. The release ZIP was downloaded publicly from a fresh clone, verified against SHA-256, and replayed with every action and selected-object position identical. Historical notes about no online publication below refer to the earlier scientific delivery.

---

# GeoPolicy Bench — version 2 progress

Status: COMPLETE V2 — experiments and delivery verification finished locally on 2026-10-01. One principal agent; no project worker remains active.

## Objective and preservation
Improve diagnosis, stable-placement scoring, view/prior comparison and reproduction. V1 is preserved at tag `geopolicy-v1-2026-09-30` (542f132): all 42 original checkpoints and 122 tracked source/config/result files are hash-verified; all 274 original HDF5 files were checked before reserved evaluation. Historical `docs/final_report.md` is unchanged. All new experiments, including negative and intermediate runs, are separate under `artifacts/v2` and `results/v2`. Branch: `codex/geopolicy-v2`. No CV was edited and nothing was published online.

## Executed study
Seven corrected 3,000-update targeted pilots, including the adaptive instruction-broadcast ACT control, selected diffusion_history and act_history_binary on validation. Disjoint confirmation gave 0/20 for each, without retuning. A matched original-data control scored 4/20 versus 1/20 with continued data; it was outside the preset ranking and does not support a continuation benefit. Ten completed intermediate main runs plus a partial checkpoint remain archived.

All 21 final models completed 8,000 updates at batch 32 over the same 200 training and 21 validation episodes: fixed/wrist/fusion crossed with prior on/off, plus selected ACT, each on three training seeds. The final protocol, checkpoints, dataset, normalization and evaluator sources were frozen before reserved-test access.

All planned final episodes completed: 3,150 main, 300 fair V1 before/after, and 360 instruction controls (3,810 total). Full raw tables are in `results/v2/raw`; all per-step traces and resume checkpoints remain local. Supervisor session 61113 exited successfully; all study stages, including delivery/report, completed with return code 0. Logs: `artifacts/v2/study_pipeline.log`, `artifacts/v2/study_logs`.

## Measured outcome
No nominal performance improvement is demonstrated. On the same new scenes and stable criterion, fusion V1 scored 16/150 versus V2 10/150 (difference −4.0 pp, descriptive 95% interval [−13.3, +4.0]); ACT V1 scored 9/150 versus V2 0/150. The targeted ACT search did not produce a reliable learned baseline. V1 remains the performance reference; V2 strengthens measurement and reproduction.

With fixed camera absent, V2 fusion scored 14/150 versus fixed 1/150 (+8.7 pp [2.7, 16.0]) and wrist-only 7/150 (+4.7 pp [−2.0, 12.0]). The wrist comparison and nominal manual-prior effect (+3.3 pp [−2.7, 9.3]) remain uncertain. No selected policy completed all four instructions on any of 30 scene-seed pairs. Report: `docs/v2_report.md`. Low loss is not treated as manipulation success.

## Verification and deliverables
- Pretest: 20 CPU tests, actual CUDA/dual-render resource gates for seven presets, exact recorded/live observations, all 221 augmented prefixes, and bit-exact continuation of the real final fusion run over its last 500 updates.
- Final audit: all 3,810 rollouts and 892,474 trace steps passed strict dwell recomputation; all frozen sources/checkpoints unchanged. XY containment uses the saved evaluator flag. Evidence: `results/v2/delivery_verification.json`.
- Compact EMA checkpoint and raw expected examples: `artifacts/v2/demo_bundle.zip`. Separate pinned-environment replay and isolated source-and-bundle-only replay both reproduced every action and selected-object position exactly, without training data or original training weights.
- Actual earliest success/failure of predeclared fusion-prior seed 0: videos in `results/v2/demo`. Video start/middle/end frames and the comparison figure were visually reviewed; success places the red cube in the blue tray, failure selects the green cube. Evidence: `results/v2/visual_verification.json`.
- Concise README, before/after report, configuration/model documentation and reproducible commands completed. Factual CV bullet proposal only: `docs/v2_cv_bullet.txt`.

## Reproduce or inspect
Run from the project root using the existing pinned environment:
`python -m geopolicy.v2 status`, `python -m geopolicy.v2 demo`, or `python -m geopolicy.v2.delivery`.
Full commands/artifact requirements: `docs/v2_reproduction.md`. The study is complete; no automatic restart is pending. Intentional new experiments require separate output paths and a new validation/test identity. The frozen V2 primary metric is retained, including continuous open-hand posture during its one-second dwell.

## Version 1 completed log

# GeoPolicy Bench — progress

Status: COMPLETE — experimental core and final verification finished locally on 2026-09-30. One principal agent; local execution only.

## Objective and authoritative context
Compare compact instruction-conditioned 3D diffusion, single fixed RGB-D versus fixed+wrist fusion, at equal demonstrations and optimization budget. Panda selection/placement in MuJoCo/robosuite; ACT baseline; real PPO expert. SmolVLA conditional on measured optimization pilot. No GitHub/Hub publication authorized.
Read the complete implementation prompt and the three referenced context files. Hardware revision supersedes historical Isaac/24–48 GB plan. CV/profile left unchanged. No applicable AGENTS.md found in target/ancestors.

## Audit
Windows: Ryzen 7 7435HS, 8 cores/16 threads; 16 GB physical RAM, ~6.3 GB initially free. C: ~65.7 GiB free. RTX 4060 Laptop 8188 MiB, ~7422 MiB initially free, 42 C, 75 W cap. Driver 591.66 exposes CUDA 13.1 compatibility (not toolkit installation). Python 3.11, 3.12, 3.13 installed. uv 0.9.29, Git available. Existing Ubuntu WSL: Python 3.12.3, ~7.6 GiB RAM limit, GPU visible. Its virtual disk reported capacity is not physical free space. No system settings altered.

## Completed
- Main Panda task, dual RGB-D geometry/action contracts, Lift smoke, lossless episode storage/reload and student information boundaries validated.
- Real BC-initialized PPO optimization verified; selected actor collected 274 traceable episodes. Failed continuation retained. Final BC and PPO controls both 82/100, with no PPO success gain demonstrated.
- The same 200 successful training and 21 validation demonstrations used for nine ACT/mono/fusion models; three seeds each, 8,000 real optimization updates each.
- Frozen primary test complete: 2,340 rollouts across five conditions, per-seed CSV/JSON and paired crossed-bootstrap intervals. Counterfactuals: 240 rollouts; secondary ablations/export: 100 rollouts.
- Authentic SmolVLA fine-tuning completed 500 optimizer updates and its final nominal test completed 20 rollouts (0/20 success). Conditional OOD deferred under the frozen zero-success gate, with measured evidence.
- Native LeRobot export reloaded exactly: 221 episodes / 21,687 frames. Trained denoiser ONNX export checked numerically and in paired closed-loop evaluation.
- Actual trained-model optimizer steps with loaded AdamW states and both RGB-D renderers alive passed for ACT, mono, fusion and SmolVLA, preserving more than 1 GiB whole-device VRAM headroom.
- Fresh pinned environment: 13 tests passed, one optional GPU test skipped by default; separate CUDA test passed. Selected validation rollout reproduced exactly. Actual last 500 CUDA training updates reproduced raw weights, EMA, optimizer, scheduler, normalization, config, progress and all RNG states bit-exactly.
- Final report, English README, dataset/model cards, contracts, licenses, manifests, figures and success/failure videos complete. Primary/new figures and both demo endpoint images visually reviewed. All 36 local documentation links resolve.
- All three environments pass dependency consistency checks. All nine selected compact checkpoint hashes, selected teacher hash and VLA adapter hash unchanged after final gates. Final 57 source fingerprints and canonical dataset manifest verified.

## Current snapshot — COMPLETE, 2026-09-30T20:58:40+02:00
No project training/evaluation/orchestration process remains active. GPU returned to approximately 7,266 MiB free at 44 C after jobs exited. No system, driver, pagefile or BIOS settings changed. No external publication or CV/profile modification occurred.

Final evidence: `docs/final_report.md`, `results/final_verification.json`, `results/benchmark_summary.json`, `results/secondary_summary.json`, `results/full_training_resume.json`, `results/combined_*_resources.json` and raw CSV/JSON under `results/raw/`. Full trajectories/checkpoints/cache remain local under ignored `artifacts/`. Publication/restoration paths are in `docs/artifact_publication.md`.

## Scientific outcome and limits
Nominal fusion 30.7% versus mono 30.3%; paired +0.3 percentage points, descriptive 95% interval [-5, +5], so no consistent nominal gain is established. Fixed-camera-loss fusion 17% versus mono 1%; +16 points [9, 23] supports a limited benefit within this scene family. Occlusion remains uncertain; depth/extrinsic cells are exploratory. ACT nominal is 0/20 for all three seeds; SmolVLA is 0/20. None of the six mono/fusion model runs completes all four instructions on any of ten counterfactual scenes. Low success, wrong-target behavior, engineered color prior and instantaneous center-proximity success constrain claims.

## Deferred scope
No required core gate remains. SmolVLA OOD is deferred by the predeclared 0/20 gate, rather than OOM. Stacking, additional tasks/seeds/language, TensorRT, physical robot transfer, Isaac and Jetson are unexecuted extensions. The executed export is an ONNX denoiser subgraph, with surrounding PyTorch preprocessing/point encoding/DDIM. Future transfer interfaces/calibration are documented without claiming hardware execution.

## Processes/checkpoints and reproducibility
Resumed pipeline session 30720 and finalizer session 86124 both exited successfully. Logs: `artifacts/remaining_pipeline_resume.log`, `artifacts/finalization_pipeline_resume.log`; complete stage flags: `artifacts/remaining_stages.json`, `artifacts/finalization_stages.json`. Historical pause inventory is preserved in `artifacts/pause_state.json`; the user's later resumption superseded it.

Principal checkpoints: `artifacts/main_runs/{mode}_s{seed}/{best,latest}.pt`; secondary checkpoints: `artifacts/secondary_runs/`; VLA adapter/base revision sidecar: `artifacts/smolvla_s0/`; selected teacher: `artifacts/teacher_selected.zip`; raw dataset: `artifacts/dataset/`. Best compact checkpoints store live training weights plus selected EMA in `extra.ema`; evaluation loads EMA and continuation restores the matching optimizer/live state. Do not alter the frozen protocol or dataset manifest to rerun a new collection; use a separate experiment checkout.

## Next action
The local repository is ready for review and optional publication when explicitly requested. The final report proposes a factual CV bullet; it was not applied. No automatic restart or further experiment is pending.

## Historical milestones (superseded process/status notes below)

## Milestone 1 — runtime passed
Native Windows stack validated: MuJoCo 3.3.7, robosuite 1.5.2, torch 2.7.1+cu128, CUDA runtime 12.8. NumPy corrected to 1.26.4 because robosuite 1.5.2 -> mink 0.0.5 requires <2. Fully installed versions in requirements-lock.txt. No upstream runtime patch required.
Lift: two 128x128 RGB-D cameras, metric depth 0.551–2.710 m, 120-step episode completes, +0.01637 m x displacement under commanded x movement. Full first-run smoke (including compilation) 39.34 s. CUDA forward/backward/AdamW step passed, torch peak allocation 17.1 MB (toy gate only, not policy memory). Report/images in artifacts/smoke_lift/.
Geometry/leakage tests: 4 passed. Main task/reference diagnostic currently running, session log artifacts/reference_pilot.log. No PPO/student training launched before its control validation.

## Milestone 2 — main task and actual PPO pilot
Scene v1 randomizes object positions AND color-to-slot assignments, plus tray positions/color slots; all four instructions change selection/goal. Script control diagnostic bootstrap: 64 episodes (artifacts/reference_pilot/bootstrap.npz); provenance scripted, not PPO.
PPO pilot: 2500 supervised bootstrap updates, then 2048 real PPO transitions. BC validation 16/20, PPO validation 17/20, parameter change L2 1.0203. This is a small pilot, not evidence of a significant gain. Files artifacts/teacher_pilot/{bc_initial,latest}.zip and validation JSON/manifests.
Main PPO continuation now running: artifacts/teacher_main.log (32768 additional transitions), periodic validation/checkpoints. CPU teacher, no rendered observations. Exact resumed physics rollout is not claimed: optimizer/RNG restored, episode reset boundary documented.
Lossless storage/re-read: one scripted diagnostic episode, 107 frames, 7.25 MB with both RGB-D and per-frame extrinsics; final endpoint saved. artifacts/data_smoke/. Student batch gate passed separately. Checkpoint test restores model/AdamW/scheduler/normalization/config/progression/all RNGs and matches next update exactly; 5 tests pass.
Next: compact ACT/CVAE and diffusion/point encoders, real optimization/memory pilots, then PPO-provenance 50-episode dataset pilot.

## Milestone 3 — policies/data gates and memory adaptation
ACT (1,051,239 params), mono/fusion diffusion (1,088,455 each) passed actual batch-4 loss/backward/AdamW/checkpoint reload/inference pilots; torch allocator peaks 45.4/59.0/72.1 MB, NOT total GPU process memory. artifacts/policy_pilots.json. No pretrained compact backbone. ACT KL weight later changed from .001 pilot setting to 1 to prevent posterior bypass; document final config.
Native LeRobot 0.4.4 isolated in .venv-vla (Python 3.11, NumPy 2.2.6); newest 0.6.1 needs Python 3.12. Main robosuite NumPy1 stack remains separate. requirements-vla-lock.txt saves resolved dependencies.
SmolVLA attempts: config factory must use base PreTrainedConfig, action chunks require explicit batch dimension; fixed. Windows error1455 during a repeated backbone load while collection was active. Concerned process exited; saved artifacts/smolvla_pilot/attempt_windows_commit.json. No pagefile/system change. Next retry after collection stops, with load_vlm_weights=False and STRICT full SmolVLA checkpoint load to avoid redundant VLM weights. Do not call this a memory-fit result or VLA training yet.
PPO continuation completed 34816 total transitions but latest validation fell to 2/20. Selected teacher remains real PPO pilot checkpoint at 2048 transitions, hash 384576a980ee55a6969d1fe00caf85cf0323dccbcf5f1fcc8562ea4d9e607d63. Negative continuation preserved, best main validation 16/20. Additional held-out validation controls: initial 0/20, script 20/20, BC 19/20, selected PPO 19/20 (no demonstrated PPO gain there).
Pilot dataset: 50 train episodes plus24 validation, actual successful counts recorded in HDF5; failures retained. Main collection of180 further episodes active in artifacts/collect_main.log. One GPU-render process, no GPU training currently active. 7 tests pass, including identical geometry under instruction changes.
Next: finish collection; serial SmolVLA step; compact student training pilot and closed-loop validation before protocol freeze/final multi-seed runs.

## Milestone 4 — data, pilots and protocol frozen
Dataset complete:250 train episodes (208 successful/42 failures),24 validation (21 successful/3 failures),274 raw episodes ~2.09 GB. First200 successful train +all21 successful validation selected identically for all students; hashes in configs/dataset_manifest.json. No test scene used. Geometry and exact checkpoint-next-step tests passed;11 tests currently pass.
SmolVLA actual memory gate passed:450.05M total/99.88M trainable parameters, batch1 loss/backward/optimizer, strict pinned base reload,2.083GB peak allocator,~5.21GB measured GPU free. Real fine-tuning500 optimizer steps/accum4 completed637s on the same200 episodes. Best validation loss.07831;2 nominal validation rollouts0/2 success, inference~424ms per2 executed actions. This is an exploratory negative result, not real-time deployment.
VLA/robosuite cohabitation solved by metadata-only OSC wheel excluding unused old Mink dependencies: all1176 runtime/assets/license files unchanged; same seed/control/point cloud identical in both environments. No system intervention.
Compact diffusion legacy epsilon pilot0/20 validation successes; sample/legacy-color-prior pilot2/20 (8/20 lifted). Revised chromatic prior (chroma40) validated by real GPU training/checkpoint/prediction gate before freeze. No guarantee of improvement. CPU2-thread DDIM policy microprofile P50~12.16ms/P95~15.20 vs CUDA~30.4/31.0ms; final inferenceCPU consistently, RTX training/profile retained.
Protocol frozen in configs/benchmark_protocol.json BEFORE main training/final tests. Main mono/fusion/ACT3 seeds,8000updates,batch32,200same demos;best EMA by offline validation only;h8 execute2. Main mono/fusion nominal/occlusion/missing-camera100paired test scenes per seed;depth/extrinsic20 exploratory. ACT20 per condition, VLA20 nominal conditional OOD. Counterfactual10scenes/four instructions; controlled50-demo data-efficiency and filter/view-dropout ablations planned after primary runs.
Active/next: serial GPU training matrix via .venv/Scripts/python scripts/run_training_matrix.py; logs/checkpoints artifacts/main_runs/. Status remains INCOMPLETE until final closed-loop evaluation, exports/figures/cards/reproduction.
Resource scheduling correction: simultaneous CPU export/teacher controls/compact validation consumed Windows commit margin (~0.55GB available despite5.5GB physical RAM free). Stopped only our auxiliary teacher-control process (PIDs18480/7620), not training; rerun controls serially. No hardware/system failure or settings change.

Remaining-stage clean_smoke: complete, wall20.1s, logartifacts\remaining_clean_smoke.log.

Remaining-stage clean_replay: complete, wall8.6s, logartifacts\remaining_clean_replay.log.

Remaining-stage profile_fusion: complete, wall6.5s, logartifacts\remaining_profile_fusion.log.

Remaining-stage profile_mono: complete, wall6.5s, logartifacts\remaining_profile_mono.log.

Remaining-stage profile_act: complete, wall3.5s, logartifacts\remaining_profile_act.log.

Remaining-stage secondary_training: complete, wall391.8s, logartifacts\remaining_secondary_training.log.

Remaining-stage main_evaluation: complete, wall10533.7s, logartifacts\remaining_main_evaluation.log.

Remaining-stage counterfactual_evaluation: complete, wall1266.1s, logartifacts\remaining_counterfactual_evaluation.log.

Remaining-stage evaluate_data50: complete, wall163.3s, logartifacts\remaining_evaluate_data50.log.

Remaining-stage evaluate_view_dropout: complete, wall318.7s, logartifacts\remaining_evaluate_view_dropout.log.

Remaining-stage evaluate_no_voxel: complete, wall159.7s, logartifacts\remaining_evaluate_no_voxel.log.

Remaining-stage evaluate_onnx: complete, wall97.0s, logartifacts\remaining_evaluate_onnx.log.

Remaining-stage smolvla_nominal: complete, wall992.7s, logartifacts\remaining_smolvla_nominal.log.

Remaining-stage summarize_primary: complete, wall5.2s, logartifacts\remaining_summarize_primary.log.

Finalizationcombined_compact: complete,logartifacts\finalize_combined_compact.log.

Finalizationcombined_vla: complete,logartifacts\finalize_combined_vla.log.

Finalizationgpu_resume: complete,logartifacts\finalize_gpu_resume.log.

Finalizationfull_training_resume: complete,logartifacts\finalize_full_training_resume.log.

Finalizationclean_tests: complete,logartifacts\finalize_clean_tests.log.

Finalizationsecondary_summary: complete,logartifacts\finalize_secondary_summary.log.

Finalizationfinal_report: complete,logartifacts\finalize_final_report.log.

V2 pilot diffusion_suffix_only/gate: complete; log `artifacts\v2\pilot_diffusion_suffix_only_gate.log`.

V2 pilot diffusion_suffix_only/train: complete; log `artifacts\v2\pilot_diffusion_suffix_only_train.log`.

V2 pilot diffusion_suffix_only/evaluate: complete; log `artifacts\v2\pilot_diffusion_suffix_only_evaluate.log`.

V2 pilot diffusion_binary/gate: complete; log `artifacts\v2\pilot_diffusion_binary_gate.log`.

V2 pilot diffusion_binary/train: complete; log `artifacts\v2\pilot_diffusion_binary_train.log`.

V2 pilot diffusion_binary/evaluate: complete; log `artifacts\v2\pilot_diffusion_binary_evaluate.log`.

V2 pilot diffusion_history/gate: complete; log `artifacts\v2\pilot_diffusion_history_gate.log`.

V2 pilot diffusion_history/train: complete; log `artifacts\v2\pilot_diffusion_history_train.log`.

V2 pilot diffusion_history/evaluate: complete; log `artifacts\v2\pilot_diffusion_history_evaluate.log`.

V2 pilot act_prior_only/gate: complete; log `artifacts\v2\pilot_act_prior_only_gate.log`.

V2 pilot act_prior_only/train: complete; log `artifacts\v2\pilot_act_prior_only_train.log`.

V2 pilot act_prior_only/evaluate: complete; log `artifacts\v2\pilot_act_prior_only_evaluate.log`.

V2 pilot act_history_binary/gate: complete; log `artifacts\v2\pilot_act_history_binary_gate.log`.

V2 pilot act_history_binary/train: complete; log `artifacts\v2\pilot_act_history_binary_train.log`.

V2 pilot act_history_binary/evaluate: complete; log `artifacts\v2\pilot_act_history_binary_evaluate.log`.

V2 pilot act_point/gate: complete; log `artifacts\v2\pilot_act_point_gate.log`.

V2 pilot act_point/train: complete; log `artifacts\v2\pilot_act_point_train.log`.

V2 pilot act_point/evaluate: complete; log `artifacts\v2\pilot_act_point_evaluate.log`.

V2 study diagnostics/input_sensitivity: complete; `artifacts\v2\study_logs\diagnostics_input_sensitivity.log`.

V2 study control/original_data_train: complete; `artifacts\v2\study_logs\control_original_data_train.log`.

V2 study control/original_data_evaluate: complete; `artifacts\v2\study_logs\control_original_data_evaluate.log`.

V2 study confirmation/diffusion: complete; `artifacts\v2\study_logs\confirmation_diffusion.log`.

V2 study confirmation/act: complete; `artifacts\v2\study_logs\confirmation_act.log`.

V2 study train/fixed_prior_s0: complete; `artifacts\v2\study_logs\train_fixed_prior_s0.log`.

V2 study train/fixed_prior_s1: complete; `artifacts\v2\study_logs\train_fixed_prior_s1.log`.

V2 study train/fixed_prior_s2: complete; `artifacts\v2\study_logs\train_fixed_prior_s2.log`.

V2 study train/fixed_no_prior_s0: complete; `artifacts\v2\study_logs\train_fixed_no_prior_s0.log`.

V2 study train/fixed_no_prior_s1: complete; `artifacts\v2\study_logs\train_fixed_no_prior_s1.log`.

V2 study train/fixed_no_prior_s2: complete; `artifacts\v2\study_logs\train_fixed_no_prior_s2.log`.

V2 study train/wrist_prior_s0: complete; `artifacts\v2\study_logs\train_wrist_prior_s0.log`.

V2 study train/wrist_prior_s1: complete; `artifacts\v2\study_logs\train_wrist_prior_s1.log`.

V2 study train/wrist_prior_s2: complete; `artifacts\v2\study_logs\train_wrist_prior_s2.log`.

V2 study train/wrist_no_prior_s0: complete; `artifacts\v2\study_logs\train_wrist_no_prior_s0.log`.

V2 pilot diffusion_suffix_only/gate: complete; log `artifacts\v2\pilot_diffusion_suffix_only_gate.log`.

V2 pilot diffusion_suffix_only/train: complete; log `artifacts\v2\pilot_diffusion_suffix_only_train.log`.

V2 pilot diffusion_suffix_only/evaluate: complete; log `artifacts\v2\pilot_diffusion_suffix_only_evaluate.log`.

V2 pilot diffusion_binary/gate: complete; log `artifacts\v2\pilot_diffusion_binary_gate.log`.

V2 pilot diffusion_binary/train: complete; log `artifacts\v2\pilot_diffusion_binary_train.log`.

V2 pilot diffusion_binary/evaluate: complete; log `artifacts\v2\pilot_diffusion_binary_evaluate.log`.

V2 pilot diffusion_history/gate: complete; log `artifacts\v2\pilot_diffusion_history_gate.log`.

V2 pilot diffusion_history/train: complete; log `artifacts\v2\pilot_diffusion_history_train.log`.

V2 pilot diffusion_history/evaluate: complete; log `artifacts\v2\pilot_diffusion_history_evaluate.log`.

V2 pilot act_prior_only/gate: complete; log `artifacts\v2\pilot_act_prior_only_gate.log`.

V2 pilot act_prior_only/train: complete; log `artifacts\v2\pilot_act_prior_only_train.log`.

V2 pilot act_prior_only/evaluate: complete; log `artifacts\v2\pilot_act_prior_only_evaluate.log`.

V2 pilot act_history_binary/gate: complete; log `artifacts\v2\pilot_act_history_binary_gate.log`.

V2 pilot act_history_binary/train: complete; log `artifacts\v2\pilot_act_history_binary_train.log`.

V2 pilot act_history_binary/evaluate: complete; log `artifacts\v2\pilot_act_history_binary_evaluate.log`.

V2 pilot act_point/gate: complete; log `artifacts\v2\pilot_act_point_gate.log`.

V2 pilot act_point/train: complete; log `artifacts\v2\pilot_act_point_train.log`.

V2 pilot act_point/evaluate: complete; log `artifacts\v2\pilot_act_point_evaluate.log`.

V2 study diagnostics/input_sensitivity: complete; `artifacts\v2\study_logs\diagnostics_input_sensitivity.log`.

V2 study control/original_data_train: complete; `artifacts\v2\study_logs\control_original_data_train.log`.

V2 study control/original_data_evaluate: complete; `artifacts\v2\study_logs\control_original_data_evaluate.log`.

V2 study confirmation/diffusion: complete; `artifacts\v2\study_logs\confirmation_diffusion.log`.

V2 study confirmation/act: complete; `artifacts\v2\study_logs\confirmation_act.log`.

V2 study train/fixed_prior_s0: complete; `artifacts\v2\study_logs\train_fixed_prior_s0.log`.

V2 study train/fixed_prior_s1: complete; `artifacts\v2\study_logs\train_fixed_prior_s1.log`.

V2 pilot act_instruction_broadcast/gate: complete; log `artifacts\v2\pilot_act_instruction_broadcast_gate.log`.

V2 pilot act_instruction_broadcast/train: complete; log `artifacts\v2\pilot_act_instruction_broadcast_train.log`.

V2 pilot act_instruction_broadcast/evaluate: complete; log `artifacts\v2\pilot_act_instruction_broadcast_evaluate.log`.

V2 study confirmation/act: complete; `artifacts\v2\study_logs\confirmation_act.log`.

V2 study train/fixed_prior_s2: complete; `artifacts\v2\study_logs\train_fixed_prior_s2.log`.

V2 study train/fixed_no_prior_s0: complete; `artifacts\v2\study_logs\train_fixed_no_prior_s0.log`.

V2 study train/fixed_no_prior_s1: complete; `artifacts\v2\study_logs\train_fixed_no_prior_s1.log`.

V2 study train/fixed_no_prior_s2: complete; `artifacts\v2\study_logs\train_fixed_no_prior_s2.log`.

V2 study train/wrist_prior_s0: complete; `artifacts\v2\study_logs\train_wrist_prior_s0.log`.

V2 study train/wrist_prior_s1: complete; `artifacts\v2\study_logs\train_wrist_prior_s1.log`.

V2 study train/wrist_prior_s2: complete; `artifacts\v2\study_logs\train_wrist_prior_s2.log`.

V2 study train/wrist_no_prior_s0: complete; `artifacts\v2\study_logs\train_wrist_no_prior_s0.log`.

V2 study train/wrist_no_prior_s1: complete; `artifacts\v2\study_logs\train_wrist_no_prior_s1.log`.

V2 study train/wrist_no_prior_s2: complete; `artifacts\v2\study_logs\train_wrist_no_prior_s2.log`.

V2 study train/fusion_prior_s0: complete; `artifacts\v2\study_logs\train_fusion_prior_s0.log`.

V2 study train/fusion_prior_s1: complete; `artifacts\v2\study_logs\train_fusion_prior_s1.log`.

V2 study train/fusion_prior_s2: complete; `artifacts\v2\study_logs\train_fusion_prior_s2.log`.

V2 study train/fusion_no_prior_s0: complete; `artifacts\v2\study_logs\train_fusion_no_prior_s0.log`.

V2 study train/fusion_no_prior_s1: complete; `artifacts\v2\study_logs\train_fusion_no_prior_s1.log`.

V2 study train/fusion_no_prior_s2: complete; `artifacts\v2\study_logs\train_fusion_no_prior_s2.log`.

V2 study train/act_selected_s0: complete; `artifacts\v2\study_logs\train_act_selected_s0.log`.

V2 study train/act_selected_s1: complete; `artifacts\v2\study_logs\train_act_selected_s1.log`.

V2 study train/act_selected_s2: complete; `artifacts\v2\study_logs\train_act_selected_s2.log`.

V2 study verification/pretest: complete; `artifacts\v2\study_logs\verification_pretest.log`.

V2 study test/fixed_prior_s0: complete; `artifacts\v2\study_logs\test_fixed_prior_s0.log`.

V2 study test/fixed_prior_s1: complete; `artifacts\v2\study_logs\test_fixed_prior_s1.log`.

V2 study test/fixed_prior_s2: complete; `artifacts\v2\study_logs\test_fixed_prior_s2.log`.

V2 study test/fixed_no_prior_s0: complete; `artifacts\v2\study_logs\test_fixed_no_prior_s0.log`.

V2 study test/fixed_no_prior_s1: complete; `artifacts\v2\study_logs\test_fixed_no_prior_s1.log`.

V2 study test/fixed_no_prior_s2: complete; `artifacts\v2\study_logs\test_fixed_no_prior_s2.log`.

V2 study test/wrist_prior_s0: complete; `artifacts\v2\study_logs\test_wrist_prior_s0.log`.

V2 study test/wrist_prior_s1: complete; `artifacts\v2\study_logs\test_wrist_prior_s1.log`.

V2 study test/wrist_prior_s2: complete; `artifacts\v2\study_logs\test_wrist_prior_s2.log`.

V2 study test/wrist_no_prior_s0: complete; `artifacts\v2\study_logs\test_wrist_no_prior_s0.log`.

V2 study test/wrist_no_prior_s1: complete; `artifacts\v2\study_logs\test_wrist_no_prior_s1.log`.

V2 study test/wrist_no_prior_s2: complete; `artifacts\v2\study_logs\test_wrist_no_prior_s2.log`.

V2 study test/fusion_prior_s0: complete; `artifacts\v2\study_logs\test_fusion_prior_s0.log`.

V2 study test/fusion_prior_s1: complete; `artifacts\v2\study_logs\test_fusion_prior_s1.log`.

V2 study test/fusion_prior_s2: complete; `artifacts\v2\study_logs\test_fusion_prior_s2.log`.

V2 study test/fusion_no_prior_s0: complete; `artifacts\v2\study_logs\test_fusion_no_prior_s0.log`.

V2 study test/fusion_no_prior_s1: complete; `artifacts\v2\study_logs\test_fusion_no_prior_s1.log`.

V2 study test/fusion_no_prior_s2: complete; `artifacts\v2\study_logs\test_fusion_no_prior_s2.log`.

V2 study test/act_selected_s0: complete; `artifacts\v2\study_logs\test_act_selected_s0.log`.

V2 study test/act_selected_s1: complete; `artifacts\v2\study_logs\test_act_selected_s1.log`.

V2 study test/act_selected_s2: complete; `artifacts\v2\study_logs\test_act_selected_s2.log`.

V2 study before_after/fusion_s0: complete; `artifacts\v2\study_logs\before_after_fusion_s0.log`.

V2 study before_after/fusion_s1: complete; `artifacts\v2\study_logs\before_after_fusion_s1.log`.

V2 study before_after/fusion_s2: complete; `artifacts\v2\study_logs\before_after_fusion_s2.log`.

V2 study before_after/act_s0: complete; `artifacts\v2\study_logs\before_after_act_s0.log`.

V2 study before_after/act_s1: complete; `artifacts\v2\study_logs\before_after_act_s1.log`.

V2 study before_after/act_s2: complete; `artifacts\v2\study_logs\before_after_act_s2.log`.

V2 study counterfactual/fusion_prior_s0_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s0_o0_g0.log`.

V2 study counterfactual/fusion_prior_s0_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s0_o0_g1.log`.

V2 study counterfactual/fusion_prior_s0_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s0_o1_g0.log`.

V2 study counterfactual/fusion_prior_s0_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s0_o1_g1.log`.

V2 study counterfactual/fusion_prior_s1_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s1_o0_g0.log`.

V2 study counterfactual/fusion_prior_s1_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s1_o0_g1.log`.

V2 study counterfactual/fusion_prior_s1_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s1_o1_g0.log`.

V2 study counterfactual/fusion_prior_s1_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s1_o1_g1.log`.

V2 study counterfactual/fusion_prior_s2_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s2_o0_g0.log`.

V2 study counterfactual/fusion_prior_s2_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s2_o0_g1.log`.

V2 study counterfactual/fusion_prior_s2_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s2_o1_g0.log`.

V2 study counterfactual/fusion_prior_s2_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_prior_s2_o1_g1.log`.

V2 study counterfactual/fusion_no_prior_s0_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s0_o0_g0.log`.

V2 study counterfactual/fusion_no_prior_s0_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s0_o0_g1.log`.

V2 study counterfactual/fusion_no_prior_s0_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s0_o1_g0.log`.

V2 study counterfactual/fusion_no_prior_s0_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s0_o1_g1.log`.

V2 study counterfactual/fusion_no_prior_s1_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s1_o0_g0.log`.

V2 study counterfactual/fusion_no_prior_s1_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s1_o0_g1.log`.

V2 study counterfactual/fusion_no_prior_s1_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s1_o1_g0.log`.

V2 study counterfactual/fusion_no_prior_s1_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s1_o1_g1.log`.

V2 study counterfactual/fusion_no_prior_s2_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s2_o0_g0.log`.

V2 study counterfactual/fusion_no_prior_s2_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s2_o0_g1.log`.

V2 study counterfactual/fusion_no_prior_s2_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s2_o1_g0.log`.

V2 study counterfactual/fusion_no_prior_s2_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_fusion_no_prior_s2_o1_g1.log`.

V2 study counterfactual/act_selected_s0_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s0_o0_g0.log`.

V2 study counterfactual/act_selected_s0_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s0_o0_g1.log`.

V2 study counterfactual/act_selected_s0_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s0_o1_g0.log`.

V2 study counterfactual/act_selected_s0_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s0_o1_g1.log`.

V2 study counterfactual/act_selected_s1_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s1_o0_g0.log`.

V2 study counterfactual/act_selected_s1_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s1_o0_g1.log`.

V2 study counterfactual/act_selected_s1_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s1_o1_g0.log`.

V2 study counterfactual/act_selected_s1_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s1_o1_g1.log`.

V2 study counterfactual/act_selected_s2_o0_g0: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s2_o0_g0.log`.

V2 study counterfactual/act_selected_s2_o0_g1: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s2_o0_g1.log`.

V2 study counterfactual/act_selected_s2_o1_g0: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s2_o1_g0.log`.

V2 study counterfactual/act_selected_s2_o1_g1: complete; `artifacts\v2\study_logs\counterfactual_act_selected_s2_o1_g1.log`.

V2 study delivery/report: complete; `artifacts\v2\study_logs\delivery_report.log`.
