# V2 reproduction and local artifacts

Run all commands from the GeoPolicy-Bench root. Validated platform: Windows,
Python 3.11, RTX 4060 Laptop 8 GB, 16 GB RAM. The existing `requirements-lock.txt`
and original runtime pins are retained; no system settings or drivers are changed.

## Short demonstration

The local `artifacts/v2/demo_bundle.zip` contains `policy.pt`, `recipe.json`,
`bundle.json`, `expected_rollouts.json`, the expected per-step success/failure
traces (when present), and a short README. The checkpoint holds
only the selected EMA parameters and normalization, with no optimizer state.
Its tensor identity against the selected training file is verified in
`configs/v2/demo_equivalence.json`. Nothing is uploaded or downloaded from a model
Hub. Restore/extract it into `artifacts/v2/demo`.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
.venv\Scripts\python -m geopolicy.v2 demo
```

This requires source, pinned environment, the tracked frozen V2 protocol and demo
equivalence metadata, and the local bundle. It requires neither training data,
original V1 weights nor optimizer checkpoints. It writes videos, traces, raw
metrics and exact-replay checks under `artifacts/v2/demo_reproduced`. Use a fresh
output folder after intentionally changing dependencies; identities prevent a
different recipe from silently reusing completed rows. Checkpoint loading is for
trusted local files only.

The demo policy is fusion with prior, training seed 0. The earliest successful
and earliest unsuccessful nominal final-test scenes are disclosed, when present.
These examples demonstrate reproducibility, not aggregate reliability. The
separate pinned `.venv-repro` verification is saved in
`results/v2/clean_reproduction.json`.

## Inspect without rerunning hours of experiments

```powershell
.venv\Scripts\python -m pytest -q -m "not gpu"
.venv\Scripts\python -m geopolicy.v2 preserve
.venv\Scripts\python -m geopolicy.v2 status
# After the study and report are complete: audit all saved final traces and demo
.venv\Scripts\python -m geopolicy.v2.delivery
```

Read `results/v2/summary.json`, the three CSV/JSON tables in `results/v2/raw`,
`results/v2/diagnostics.json`, `results/v2/selection.json`, and
`results/v2/pretest_verification.json`. The tracked raw tables contain every final
success and failure. Per-step traces, training curves, resource samples,
optimizers and checkpoints live locally under ignored `artifacts/v2`.
The delivery audit needs those local traces and checkpoints. It independently
recomputes the strict dwell from saved signals, checks frozen source/checkpoint
identities and the compact ZIP, and writes `results/v2/delivery_verification.json`.
XY containment uses the saved evaluator flag; this is an evidence consistency
check, not an independent simulator geometry implementation.

## Resume the executed study

Check running Python processes and pipeline logs before starting a second job.
The supervisors are serial and preserve completed stages; they do not coordinate
two independently launched supervisors. Run one supervisor and one GPU-heavy
child at a time.

```powershell
.venv\Scripts\python -m geopolicy.v2 augment
.venv\Scripts\python -m geopolicy.v2 pilots
.venv\Scripts\python -m geopolicy.v2 study
```

These commands require all preserved V1 weights, the original 274 HDF5 episodes,
the selected PPO teacher, and the V2 dataset/checkpoints for completed stages.
`augment` reads original files, skips completed V2 episodes, records 30 real
post-release PPO-teacher actions and retains original normalization.
`pilots` executes the six initial diagnostic recipes and the additional adaptive
ACT instruction-broadcast control. That seventh hypothesis was added after the
initial validation diagnostics, before the reserved test, without changing the
ranking rule; its negative result is retained.
`study` selects only from validation, reports independent confirmation, trains
21 main models, verifies real recovery, freezes the final protocol, and evaluates
the registered checkpoints. It writes results and the local bundle. Stage state
is `artifacts/v2/study_stages.json`; failures preserve completed rows and the last
500-update checkpoint. A resumed job must use an identical run config.

```powershell
# Example: resume one interrupted main run, with its exact config
.venv\Scripts\python -m geopolicy.v2 train --run-config configs/v2/runs/fusion_prior_s0.json --out artifacts/v2/main_runs/fusion_prior_s0 --resume artifacts/v2/main_runs/fusion_prior_s0/latest.pt
# Example: reproduce the already frozen nominal test in a new output directory
.venv\Scripts\python -m geopolicy.v2 evaluate --checkpoint artifacts/v2/main_runs/fusion_prior_s0/best.pt --out artifacts/v2/replayed_test --first-seed 300000 --episodes 50 --conditions nominal --reserved-test
```

Do not overwrite published local measurements with a new experiment. A fresh
teacher collection or intentionally changed recipe needs a separate experiment
directory/checkout, manifest, validation selection and reserved test protocol.
Frozen checkpoint and evaluator hashes prevent silently evaluating changed
weights or code under V2's test identity.

## Configuration and information contract

`configs/v2/recipe.json` centralizes all important budgets, dimensions, stability
thresholds, resource limits and scene ranges. `presets.py` names the seven targeted
hypotheses; `configs/v2/act_diagnostic.json` records the adaptive ACT rationale
and its fresh confirmation seeds. `configs/v2/runs` records every main network and seed.
`diagnostic_recipe.json` preserves the original-data diagnostic identity;
`original_data_recipe.json` specifies the matched dataset-continuation control.

Training uses 200 successful demonstrations and 21 separate recorded validation
episodes. In the augmented manifest, fields inherited from V1 (including
`frames`, `seconds` and terminal V1 metrics) describe the original prefix.
`original_frames` plus `post_release_frames` gives the V2 total; each HDF5
`metadata.frames` and actual dataset length contain that total. Final-image fields
are updated to the end of the continuation. Original final RGB/depth is verified
against the first recorded continuation frame; all other prefix arrays match
exactly. The suffix is the preserved actual PPO actor, not relabeled scripted BC.

State inputs are a robot-only allowlist, optionally with four causal snapshots.
ACT uses RGB; diffusion uses calibrated XYZRGB, including named-camera selection.
Instruction input is four semantic one-hot tokens. Ground-truth cube and tray
states are confined to teacher generation and evaluation. The fixed-camera-loss
condition preserves the wrist-only observation. All point policies have a total
512-point budget, independent of view count. The prior ablation removes manual
chromatic attention bias; it retains RGB and learned attention/moments.

Placement scoring is documented in `docs/v2_design.md` and tested in
`tests/v2/test_stability.py`. Policy actions continue during its one-second dwell.
No evaluator correction or phase-dependent action is applied to a student.

## Resources and preservation

Each pilot runs an actual batch-32 AdamW update, simultaneous dual-camera render
and exact next-update recovery before its training budget. Main optimization uses
CUDA FP32; evaluation uses CPU with two Torch threads and GPU rendering. Logs
distinguish Torch allocated/reserved memory, device-wide memory, temperature,
Windows committed memory, process private/RSS and disk space. Persistent unsafe
samples stop the worker. Default checkpoints every 500 updates support recovery;
episodes are atomically saved after each rollout. Do not launch parallel Torch
workers on this 16 GB machine.

V1 tag: `geopolicy-v1-2026-09-30`, commit `542f132`. Original results, data,
checkpoints and code are hash-checked against `configs/v2/v1_inventory.json` and
the original dataset manifest. The old README is in `docs/v1`; the historical
`docs/final_report.md` is unchanged. New results and negative experiments stay
separate under `results/v2` and `artifacts/v2`. No CV is modified and no online
publication is performed.
