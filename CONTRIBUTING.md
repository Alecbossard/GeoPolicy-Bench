# Contributing

GeoPolicy-Bench is a small simulation research project. Useful contributions improve manipulation, diagnose a failure mechanism, or strengthen a controlled comparison.

## Local checks

Use Python 3.11 and the environment described in the [README](README.md).

```powershell
.venv\Scripts\python -m pytest -q -m "not simulation and not gpu"
.venv\Scripts\python scripts/check_publication.py
```

GitHub Actions runs CPU contracts and publication evidence/link checks on Windows. It does not train policies, render the demo or validate real-robot behavior. The executed local GPU and full-trace checks are recorded under `results/v2/`.

## New experiments

Preserve the V1/V2 snapshots and their raw outcomes. Use separate configurations, artifact folders and result folders for new experiments; keep unsuccessful runs. The frozen V2 evaluator and checkpoint identities must remain unchanged for V2 reproduction.

Select changes on validation. Reserve fresh test scenes before final evaluation, repeat principal comparisons over training seeds, and keep demonstrations, actions and budgets matched. Report absolute success and failure stages; a loss reduction alone is not a manipulation improvement. State whether an explanation is a hypothesis, a diagnostic association or an isolated causal result.

Run one heavy GPU worker at a time on the validated laptop. Keep resume checkpoints and measure resource headroom before long jobs. Include the command, configuration, data identity, validation choice and raw results with an experimental change.

The root README is maintained as a project presentation. The report command writes its numerical README draft to ignored `artifacts/v2/generated_README.md`, so regenerating a report does not replace the curated front page.

## Scope

The compact ACT/diffusion implementations are independent adaptations. Credit upstream projects and retain [third-party notices](docs/THIRD_PARTY_NOTICES.md). Physical hardware, broader language grounding and new tasks require their own evaluation; they are not established by the current simulation results.
