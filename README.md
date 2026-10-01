# GeoPolicy-Bench

**Question:** does calibrated fixed+wrist RGB-D fusion improve instruction-conditioned Panda placement beyond either camera alone, and how much comes from a manual color prior?

**Measured V2:** fusion achieved **10/150 (6.7%) stable nominal placements** over three training seeds and 50 reserved scenes. Under missing fixed camera, fusion minus fixed was **+8.7 pp**, descriptive 95% interval **[+2.7, +16.0]**. This is a compact simulation study with limited reliability; a positive point estimate alone does not establish superiority. The selected compact ACT achieved 0/150 stable nominal placements.

V2 strengthens verification and reproduction without demonstrating a nominal performance gain: preserved V1 fusion scored 16/150 on the same new stable criterion. The targeted ACT search did not yield a reliable learned baseline. V1 remains preserved as the performance reference.

| Policy | Nominal stable /150 | Fixed camera absent: stable /150 |
| --- | --- | --- |
| Fixed RGB-D + prior | 10 | 1 |
| Wrist RGB-D + prior | 7 | 7 |
| Fusion RGB-D + prior | 10 | 14 |
| Selected compact RGB ACT | 0 | 1 |

With the fixed camera absent, fusion minus wrist-only was +4.7 pp [-2.0, +12.0]. For nominal fusion, manual prior on minus off was +3.3 pp [-2.7, +9.3]. These descriptive 95% intervals distinguish the surviving wrist sensor from fusion and the manual prior; all ablations and failures are in the report.

[Success video](results/v2/demo/success/nominal_300012_stable.mp4) · [Failure video](results/v2/demo/failure/nominal_300000_failure.mp4) — actual disclosed test episodes, local checkpoint replay verified in a separate pinned environment.

![Stable placement comparison](results/v2/figures/main_comparison.png)

## Reproduce the short local demo

Validated on Windows, Python 3.11, RTX 4060 Laptop 8 GB. From the project root:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
# Restore/extract the local artifacts/v2/demo_bundle.zip into artifacts/v2/demo
.venv\Scripts\python -m geopolicy.v2 demo
```

No training data is needed for the demo. Weights stay local under ignored `artifacts/`; the Git repository alone does not contain them. Raw expected actions/metrics and hashes accompany the local bundle. [Full commands and exact artifact requirements](docs/v2_reproduction.md), [measured before/after report](docs/v2_report.md), [raw test CSV](results/v2/raw/main_rollouts.csv), [configuration](configs/v2/recipe.json), [progress](PROGRESS.md).

## What is controlled

Six diffusion variants: fixed, wrist and fusion, each with/without the chromatic prior. Same 200 successful demonstration prefixes plus 30 recorded PPO-teacher steps after release, 21 validation episodes, frozen normalization, 7D actions, 512 points, 8,000 updates, batch 32, three training seeds. Seven targeted 3,000-update pilots select the recipe on validation; a new frozen test uses 50 shared scenes in nominal, fixed occlusion and fixed-camera-loss conditions. ACT is a compact RGB adaptation with different capacity; a rejected 3D ACT-style pilot is reported separately.

The strict success metric requires full contained placement, low object speed, and fingers kept open without finger/object contact for one continuous second while the policy keeps acting. Closing the empty hand during that dwell can fail this conservative score; such a failure alone does not prove the object moved. Failure traces distinguish target selection, grasp, transport, release and post-release stability. Normalization, rendering alignment, physical gripper sign and real checkpoint continuation were verified. Per-frame loss is not evidence of manipulation success.

## Before/after and limits

On the same new nominal test and stable criterion, preserved V1 fusion scored 16/150 versus V2 10/150. The V2 recipe changes multiple components; this does not identify one causal improvement. V1 historical instantaneous scores remain intact at tag `geopolicy-v1-2026-09-30`, with its [original README](docs/v1/README_original.md) and [original report](docs/final_report.md). All new experiments are separate under `artifacts/v2` and `results/v2`.

Known calibrated simulation, two colored cubes, two receptacles and an instruction vector containing two one-hot pairs; no pretrained vision, real-robot transfer, open-vocabulary grounding or official ACT/DP3 reproduction claimed. A scripted phase curriculum initializes a real PPO teacher; its historical PPO update showed no success gain over BC. PPO and SmolVLA remain secondary V1 extensions. Three seeds, narrow scenes, negative outcomes and residual failures limit the conclusions. Nothing was published online and the user's CV was not edited.

[Architecture and sources](docs/architecture.md) · [V2 design](docs/v2_design.md) · [Third-party notices](docs/THIRD_PARTY_NOTICES.md) · [Factual CV bullet proposal](docs/v2_cv_bullet.txt)
