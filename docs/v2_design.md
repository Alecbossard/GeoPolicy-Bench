# Version 2: diagnostic plan and information boundaries

Version 1 is preserved at Git tag `geopolicy-v1-2026-09-30` (commit `542f132`). Its
raw results, trained weights, dataset and historical report are retained. New
experiments write only to `artifacts/v2` and `results/v2`. The original README and
report are copied to `docs/v1`; the historical `docs/final_report.md` stays intact.

First measure recorded-frame preprocessing, action normalization, physical
gripper sign, ACT posterior/prior prediction differences and closed-loop failure
stages. Future-action-conditioned posterior predictions are an offline oracle
diagnostic, never an evaluation policy. Zero latent at inference is standard in
[original ACT](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)
and [LeRobot ACT 0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/modeling_act.py);
its presence alone is not a demonstrated implementation bug.

The Panda gripper converts the command's sign to an incremental binary open/close
target. Small positive magnitudes do not mean a weak closing command. Diagnose
sign mistakes and repeated sign switching before attributing failures to weak grip.

V2 placement requires the lifted selected cube's rotated XY corners inside the
chosen tray, fingers open at least 45 mm with no finger/object contact, linear
speed at most 0.02 m/s and angular speed at most 0.25 rad/s, for samples spanning
one continuous second. At 20 Hz this requires at least 21 consecutive valid
samples. The policy continues issuing actions during verification; no evaluator
waypoint, object-aware correction or assisted release is applied. The historical
instantaneous score is also logged separately. Stage classifications are
measured geometric/contact proxies, not access to the teacher's phase variable.

`configs/v2/recipe.json` centralizes dimensions, budgets, limits and scene ranges.
All compact variants use exactly the same 200 successful V1 training episodes,
normalization, 7D action contract and update budget. Planned view/prior controls:
fixed only, wrist only, fusion; chromatic prior present/absent. A missing fixed
camera must not remove the wrist from a wrist-only policy.

Pilot recipe selection uses validation seeds 100200–100219, followed by separate
confirmation seeds 100240–100259. Final scenes start at 300000, disjoint from V1
tests, and must not run before a final protocol and selected recipes are frozen.
Main comparisons retain three training seeds. No selection on final test results.
PPO and SmolVLA remain historical extensions unless these diagnostics justify work.
