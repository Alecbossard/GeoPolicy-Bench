# GeoPolicy Bench — progress

Status: PAUSED at the explicit user request on 2026-09-30; benchmark INCOMPLETE. One principal agent; local execution only.

## Objective and authoritative context
Compare compact instruction-conditioned 3D diffusion, single fixed RGB-D versus fixed+wrist fusion, at equal demonstrations and optimization budget. Panda selection/placement in MuJoCo/robosuite; ACT baseline; real PPO expert. SmolVLA conditional on measured optimization pilot. No GitHub/Hub publication authorized.
Read the complete implementation prompt and the three referenced context files. Hardware revision supersedes historical Isaac/24–48 GB plan. CV/profile left unchanged. No applicable AGENTS.md found in target/ancestors.

## Audit
Windows: Ryzen 7 7435HS, 8 cores/16 threads; 16 GB physical RAM, ~6.3 GB initially free. C: ~65.7 GiB free. RTX 4060 Laptop 8188 MiB, ~7422 MiB initially free, 42 C, 75 W cap. Driver 591.66 exposes CUDA 13.1 compatibility (not toolkit installation). Python 3.11, 3.12, 3.13 installed. uv 0.9.29, Git available. Existing Ubuntu WSL: Python 3.12.3, ~7.6 GiB RAM limit, GPU visible. Its virtual disk reported capacity is not physical free space. No system settings altered.

## Completed
- Context read; project directory created (previously absent).
- Initial hardware audit and official-source research.

## Current snapshot — PAUSED, 2026-09-30T19:26:18+02:00
The user requested pause. All project evaluation/orchestration processes have been stopped; no automatic resumption is scheduled. Completed checkpoints and atomic rollout files are preserved. No serious PC issue caused this pause.

All nine primary students finished training. The 2,340 primary rollouts, 240 counterfactual rollouts and 100 secondary ablation/export rollouts are complete. SmolVLA fine-tuning is complete; its nominal test is partial: 10/20 completed scenes, 0 successes so far. These partial results are not a final test conclusion. The interrupted episode will restart; completed scenes are skipped.

Primary result tables and plots exist. Final combined memory gates, exact full-training resume verification, final clean test suite, secondary summary, final report and final artifact review remain pending. The project has not reached its completion criteria. CV/profile and external publication remain unchanged.

Pause inventory: `artifacts/pause_state.json`. Completed stage flags: `artifacts/remaining_stages.json`. On explicit user resumption, run the following from the project directory in separate processes; the finalizer waits for the remaining pipeline:

```powershell
.venv\Scripts\python.exe scripts/run_remaining.py
.venv\Scripts\python.exe scripts/finish_after_pipeline.py
```

The pipeline skips completed stages and SmolVLA skips saved scenes. Keep the same checkpoints, pinned base revisions, evaluation identity and frozen protocol. No training or evaluation is running while paused.

## Remaining gates
1. Finish frozen main student closed-loop test (mono/fusion3seeds,100 principal cells/20 exploratory),ACT20cells,counterfactuals,SmolVLA20nominal,secondary ablations and ONNXpairedclosed-loop.
2. Combined optimizer+dual-renderer resource gates,actual CUDA exact-next-update resume,final clean tests and aggregate figures/report.
3. Final visual QA,model cards/README/source hashes/local Git audit;no external publication.

## Active processes/checkpoints/results
Remaining pipeline exec session34403,artifacts/remaining_pipeline.log and remaining_stages.json. Active stage main_evaluation,artifacts/remaining_main_evaluation.log;per-cell artifacts/final_evaluations/{mode}_s{seed}/{condition}/rollouts.json+CSV+identity+execution.log;checkpoint hashes match. Finalization exec session2641 waits for all experimental stages in artifacts/finalization_pipeline.log;resumable final gates use artifacts/finalization_stages.json. Old waiting session81756 was stopped cleanly before any child started and replaced to include actual500-update full-training replay. Principal training session11067 finished;all checkpoints artifacts/main_runs/{mode}_s{seed}/{best,latest}.pt. All secondary checkpoints artifacts/secondary_runs/. Data/weights remain ignored. Resource guards monitor temperature/disk and training/VLA commit;normal observed temperatures~43–53C. Pipeline serially performs remaining main/counterfactual/ablation/VLA experiments. Do not launch a second heavy GPU job.

## Next action
Monitor serial remaining pipeline and finalization. Report generator scripts/write_final_report.py consumes completed primary/secondary/resource artifacts and refuses incomplete gates. Publication/replay instructions are in docs/artifact_publication.md. Every completed test episode is persisted and resume identity verified. Never adjust the frozen primary recipe using final test outcomes. Final opt-in CUDA exact-next-update test runs AFTER heavy pipeline jobs complete. Finish visual QA/documentation/source fingerprints/Git only after actual results exist.

Additional final gate: scripts/verify_full_training_resume.py replays mono seed1 from selected update7,500 to8,000 in a disposable copy and compares raw weights/EMA/optimizer/scheduler/normalization/config/progress/all RNGs with the original latest checkpoint. It is queued after VLA/GPU gates. Teacher reward/phases and a clearly untested reward-design hypothesis are documented in docs/teacher_recipe.md. VLA inference discards unused AdamW CPU storage while loading the authentic base; selected checkpoint files are unchanged.

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
