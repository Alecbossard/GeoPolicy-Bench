# Local artifacts and optional publication

The repository is local. No GitHub repository, Hub dataset or model has been published. Publication requires the owner's explicit request. Data, checkpoints, caches and virtual environments are excluded from Git; the repository includes lightweight metrics, figures and success/failure demos.

## Artifact map

| Local path | Contents | Role |
| --- | --- | --- |
| `artifacts/dataset/` | 274 lossless HDF5 episodes, including failures | Main source collection; `configs/dataset_manifest.json` records hashes and selected episode IDs |
| `artifacts/teacher_selected.zip` | Selected BC-initialized PPO checkpoint | Actual demonstration actor; selected at 2,048 PPO transitions |
| `artifacts/teacher_pilot/` | Initial/BC/PPO controls, optimizer and RNG artifacts | Provenance and teacher learning |
| `artifacts/teacher_main/` | Negative continuation to 34,816 total transitions | Diagnostic; never selected for data |
| `artifacts/main_runs/{mono,fusion,act}_s{0,1,2}/` | Best/latest checkpoints, manifests, learning curves and resources | Nine principal models; best selected only by offline validation |
| `artifacts/secondary_runs/` | 50-demo and view-dropout models | Predeclared single-seed ablations |
| `artifacts/smolvla_s0/` | Trainable adapter, AdamW/scheduler/RNG/statistics, curves and manifest | Actual 500-update fine-tuning; requires pinned base |
| `artifacts/lerobot_dataset/` | Reloaded native LeRobot RGB/state/action/text export and source sidecar index | Compatible data distribution; depth remains lossless HDF5 |
| `artifacts/final_evaluations/` | Per-cell raw rollouts, evaluation identity, execution logs and sample videos | Primary closed-loop evidence |
| `artifacts/counterfactual_evaluations/` | Four instructions per identical scene | Instruction-dependent behavior and initial sensor hashes |
| `artifacts/secondary_evaluations/`, `artifacts/smolvla_test*/` | Paired ablations and VLA evaluation | Exploratory results |
| `artifacts/onnx_export/denoiser.onnx` | Trained FP32 denoiser subgraph | Partial deployment export; same point encoder/DDIM retained |
| `results/` | Aggregated metrics, checkpoint audits, resource profiles, figures and demos | Lightweight reviewable evidence committed locally |

`best.pt` holds the live training weights under `model`, selected EMA weights under `extra.ema`, and full optimizer/scheduler/RNG state at that update. Evaluation loads `extra.ema`; training continuation restores the matching live weights and EMA separately. Use `latest.pt` for ordinary continuation. Changing the total update budget on resume is rejected rather than silently changing the cosine schedule. Retraining a different recipe requires a separate output directory and config.

## Reproduce with preserved artifacts

Follow the pinned installation instructions in the README, restore data/weights at the relative paths above, and run from the repository root. The Windows target uses native WGL; other platforms are not claimed tested.

```powershell
.venv\Scripts\python scripts/audit_dataset.py --verify-frozen
.venv\Scripts\python scripts/audit_checkpoints.py
.venv\Scripts\python -m pytest -q
# GPU test is explicitly opt-in; run without another heavy GPU job.
$env:GEO_GPU_TESTS = '1'
.venv\Scripts\python -m pytest tests/test_gpu_resume.py -q
Remove-Item Env:\GEO_GPU_TESTS

# A deterministic replay; separate output keeps original evidence intact.
.venv\Scripts\python -m geopolicy.cli evaluate --checkpoint artifacts/main_runs/fusion_s0/best.pt --out artifacts/reproduce_success --episodes 1 --first-seed 200001 --conditions nominal --device cpu --execute-steps 2 --videos 1
.venv\Scripts\python scripts/make_demos.py
.venv\Scripts\python scripts/summarize_results.py
.venv\Scripts\python scripts/summarize_secondary.py
```

The full frozen evaluation matrix resumes complete scene rows only when checkpoint hash, action execution, device and sensor settings match. It loads the checkpoint on each cell and skips existing rows; remove nothing if an interruption occurs. To repeat independent wall-clock measurements, use a new output path instead of averaging cached measurements with new ones. Episode seed/state replay was verified, not a claim of identical wall times or compatibility with untested driver versions.

The VLA adapter must be loaded with `scripts/vla_common.py::restore_adapter`, not as a stand-alone full foundation model. Base model/tokenizer/config revisions are pinned in `configs/pretrained_revisions.json`. The original adapter was trained before revision fields were added to the checkpoint format; `artifacts/smolvla_s0/base_revisions.json` preserves its original pinned recipe without changing any trained tensors. This sidecar is mandatory when distributing that adapter. Future checkpoints embed these fields directly.

Rebuild the metadata-only OSC wheel before installing `requirements-vla-lock.txt`. Its manifest verifies official wheel provenance and all unchanged runtime/assets/license files. It is validated for this Panda OSC task, and does not provide Mink whole-body IK compatibility. The base model can be downloaded from its pinned upstream snapshot; do not distribute the entire local Hugging Face cache or authentication files.

## Prepare an archive when publication is requested

Include the dataset/model cards, MIT project license, upstream third-party notices, frozen configs, episode/checkpoint hashes and actual run manifests. Preserve the relative source HDF5 paths referenced by the LeRobot sidecar, or rewrite and validate that index for the chosen archive layout. Label unsuccessful episodes, bootstrap script data and negative checkpoints correctly. Adapter publication must credit the Apache-2.0 SmolVLA/LeRobot base and include its pinned revision sidecar. Keep virtual environments, caches, Windows absolute paths and tokens out of upload archives.

The proposed CV bullet in the final report describes executed simulation work only. No CV/profile edits are part of this repository.
