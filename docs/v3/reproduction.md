# V3 local commands

Run from the GeoPolicy-Bench root using the existing pinned environment.
The public V2 README and all historical scientific artifacts stay unchanged.

```powershell
.venv\Scripts\python -m geopolicy.v3 preserve
.venv\Scripts\python -m geopolicy.v3 checks
.venv\Scripts\python -m geopolicy.v3 baseline
```

The supervisor launches one child at a time, preserves completed stages, and
resumes incomplete training from `latest.pt`. Every 500 updates records model,
optimizer, scheduler, EMA, normalization, RNG and batch RNG. Evaluation stores
completed rows and full traces after each episode. An identity mismatch requires
a distinct V3 experiment name; it never overwrites historical results.

Example diagnostic replay, with an existing local checkpoint:

```powershell
.venv\Scripts\python -m geopolicy.v3 evaluate --name local_review --checkpoint artifacts/v3/runs/diffusion_overfit_s0/best.pt --first 10000 --episodes 4 --videos 1
```

This example uses training scenes and therefore is an overfit diagnostic.
The fresh 400000–400049 test remains locked. No V3 final performance result or
functional multi-object baseline is claimed at this stage.

Source and configs: [plan](../../configs/v3/plan.json),
[progress](PROGRESS.md), [input/controller/CUDA checks](../../results/v3/initial_checks.json).
