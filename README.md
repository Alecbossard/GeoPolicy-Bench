# GeoPolicy Bench

**Work in progress — incomplete benchmark.** Local Panda selection/placement benchmark for instruction-conditioned RGB-D policies. The main question is whether calibrated fixed+wrist RGB-D fusion improves robustness over one fixed camera at equal demonstrations, architecture and optimization budget. A negative result is publishable; no performance gain is presumed.

This repository contains a MuJoCo/robosuite task, an explicitly BC-initialized PPO teacher, lossless trajectory storage, compact ACT and DP3-inspired diffusion adaptations, and closed-loop evaluation. An authentic SmolVLA optimization pilot passed and500-step fine-tuning completed; its exploratory behavior is reported separately. No exact DP3/ACT paper reproduction, official LIBERO score, physical robot transfer or Isaac Lab execution is claimed. Current status and measured milestones: [PROGRESS.md](PROGRESS.md).

## Local installation

Validated target: Windows, Python 3.11, RTX 4060 Laptop 8 GB. Drivers remain unchanged. Install in an isolated environment:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
uv pip install --python .venv\Scripts\python.exe -e .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\smoke_lift.py
```

The lock file excludes machine-specific editable paths. NumPy1.26.4 is required by the chosen robosuite/mink dependency chain. A clean `.venv-repro` installed these pins and passed12 tests locally. LeRobot uses a separate `.venv-vla` with NumPy2; its metadata-only OSC simulation wheel and compatibility proof are documented below. See [technical decisions](docs/technical_decisions.md) and [contracts](docs/contracts.md).

## Commands

```powershell
# Scripted diagnostic/bootstrap, never labeled PPO demonstrations
.venv\Scripts\python scripts\reference_pilot.py --episodes 64
# Actual BC-initialized on-policy PPO pilot and continuation
.venv\Scripts\python -m geopolicy.cli teacher-train --out artifacts/teacher_pilot --steps 2048
.venv\Scripts\python -m geopolicy.cli teacher-train --out artifacts/teacher_main --steps 32768 --resume artifacts/teacher_pilot/latest.zip
# Actual selected teacher was the2048-step pilot, chosen on validation
Copy-Item artifacts/teacher_pilot/latest.zip artifacts/teacher_selected.zip
# Lossless collection with explicit checkpoint hash/provenance
.venv\Scripts\python -m geopolicy.cli collect --out artifacts/dataset --episodes 250 --first-seed 1000 --teacher artifacts/teacher_selected.zip
.venv\Scripts\python -m geopolicy.cli collect --out artifacts/dataset --episodes 24 --first-seed 100000 --teacher artifacts/teacher_selected.zip
# Compact student training (same budget for mono and fusion)
.venv\Scripts\python scripts/run_training_matrix.py
# Optimizer/scheduler/normalization/RNG/batch-generator resume
.venv\Scripts\python -m geopolicy.cli student-train --mode fusion --out artifacts/main_runs/fusion_s0 --seed 0 --updates 8000 --limit 200 --batch-size 32 --device cuda --resume artifacts/main_runs/fusion_s0/latest.pt
# Validation-only behavior checks; final test requires frozen protocol
.venv\Scripts\python -m geopolicy.cli evaluate --checkpoint artifacts/main_runs/fusion_s0/best.pt --out artifacts/validation_fusion --episodes 20 --first-seed 100100 --conditions nominal --videos 2 --device cpu --execute-steps 2
# Complete all nine main trainings before running the frozen final test
.venv\Scripts\python scripts/run_evaluation_matrix.py
.venv\Scripts\python scripts/run_evaluation_matrix.py --stage counterfactual
.venv\Scripts\python scripts/summarize_results.py
# Predefined secondary training and deployment export
.venv\Scripts\python scripts/run_secondary_training.py
.venv\Scripts\python scripts/export_and_check.py
```

Run one GPU training/inference job at a time and prefer serial rendering/model loading on this 16 GB RAM PC. Windows GPU contexts consume committed host memory as well as VRAM. Store data/checkpoints under ignored `artifacts/`; do not commit secrets or model/data blobs. No external publication is performed by these commands.

## Structure and adaptations

`src/geopolicy/`: runtime, task, sensors, data, PPO teacher, policies, training, evaluation, checkpoint and CLI. `tests/`: geometry, validity masks, split boundaries, instruction counterfactual scene identity, leakage allowlist and exact optimization resume. `scripts/`: runtime/model pilots and controls. `docs/`: decisions and contracts. `artifacts/`: local measured runs, raw trajectories, models, images, videos and reports (ignored).

ACT adaptation: random compact CNN, CVAE latent encoder and Transformer action-chunk decoder, explicit four-dimensional semantic instruction tokens. 3D diffusion adaptation: XYZRGB point encoder with learned attention/max pooling and a declared RGB chromatic prior for four known colors, instruction/proprioception conditioning, temporal FiLM residual denoiser, clean-action prediction/cosine DDPM training and10-step DDIM inference. The two diffusion variants differ only in input views and fusion; fixed512-point budget. No pretrained backbone in these compact policies. SmolVLA uses its authentic pretrained backbone/action model and official preprocessing.

## LeRobot and SmolVLA environment

```powershell
uv venv --python 3.11 .venv-vla
.venv\Scripts\python scripts/build_osc_sim_wheel.py
uv pip install --python .venv-vla\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-vla\Scripts\python.exe -r requirements-vla-lock.txt
.venv-vla\Scripts\python scripts/export_lerobot.py
.venv-vla\Scripts\python scripts/smolvla_pilot.py
.venv-vla\Scripts\python scripts/smolvla_train.py --updates 500 --accumulation 4
.venv-vla\Scripts\python scripts/smolvla_train.py --updates 500 --accumulation 4 --resume artifacts/smolvla_s0/latest.pt
.venv-vla\Scripts\python scripts/smolvla_evaluate.py --episodes 20 --first-seed 200000 --out artifacts/smolvla_test --conditions nominal
```

The custom wheel changes dependency metadata only, removing unused Mink/qpsolvers requirements; all1176 runtime/assets/license files are byte-identical to upstream. It supports the validated Panda OSC path. Both environments produced identical scene geometry, control response and RGB-D point clouds. This preserves one-process VLA+rendering without merging incompatible NumPy requirements.

The native LeRobot export contains221 episodes/21,687 frames with lossless PNG RGB, Parquet state/action and actual task text. Reloaded RGB/state/actions match source exactly. `depth_point_sidecars.json` maps each LeRobot episode/frame to its source HDF5 (lossless depth, points, transforms, timestamps and hashes). No Hub publication occurs. Main HDF5 failures remain available separately.

Teacher limitation: a scripted privileged phase machine supplies curriculum waypoints; supervised BC initializes the low-level actor before PPO optimization. PPO is real, but task sequencing is not learned end to end. Distinguish random, scripted, BC-only and PPO controls. Students never see teacher phases, waypoint errors or object truth poses.

## Current limitations

The instruction vocabulary, rigid objects and scene family are deliberately narrow. Simulation calibration is known; sensor corruptions are controlled approximations. No arbitrary language grounding, unseen geometry, autonomous camera calibration or sim-to-real success is established. A separate transfer interface/calibration procedure and final dataset/model cards will accompany completed experiments.

## Sources and credits

[MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0), [robosuite](https://github.com/ARISE-Initiative/robosuite) (MIT with upstream notices), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT), [LeRobot/ACT/SmolVLA](https://github.com/huggingface/lerobot) (Apache-2.0), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy) (MIT). Our compact models are independent adaptations inspired by these sources. See [third-party notices](docs/THIRD_PARTY_NOTICES.md).
