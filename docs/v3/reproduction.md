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

## Current resume point (3 October)

The interaction study has completed. Its distinct confirmation stopped after
10 saved scenes because Windows free commit stayed below 1 GiB for three samples.
The evaluation worker consumes about 3.5 GiB private memory. Free roughly 6 GiB
of Windows commit before resuming; the resource thresholds must remain enabled.
The lightweight preflight now checks this before importing CUDA-enabled Torch,
using `configs/v3/runtime_limits.json`; it does not change the scientific plan
or any recorded evaluation identity.

```powershell
.venv\Scripts\python -m geopolicy.v3 interaction-confirmation
```

This preserves the completed rows and evaluates only remaining scenes. It writes
`configs/v3/selection.json` only after all three seeds are complete and the
predeclared selection rule has been checked. The partial 10/10 is not a 20/20
confirmation and does not open the next gate.

Then, in order, using the confirmed selection:

```powershell
.venv\Scripts\python -m geopolicy.v3 progressive --task two_objects_one_goal
.venv\Scripts\python -m geopolicy.v3 progressive --task full
```

Each task starts with collection and a seed-0 pilot; failure stops expansion.
The full task requires a passed three-seed tuning/confirmation gate on the
two-object/one-goal task. Neither command unlocks the fresh final test.

Read-only trace reconstruction and lightweight validation-gate checks:

```powershell
.venv\Scripts\python -m geopolicy.v3 audit
.venv\Scripts\python -m unittest discover -s tests/v3 -p test_validation_gates.py -v
.venv\Scripts\python -m unittest discover -s tests/v3 -p test_resource_preflight.py -v
```
