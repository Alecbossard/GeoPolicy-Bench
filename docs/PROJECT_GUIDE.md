# Project guide

## Review in a few minutes

1. Read the [README](../README.md) and watch both disclosed V2 examples.
2. Read the [before/after report](v2_report.md), especially the ACT component interventions, negative data-continuation control and paired view/prior comparisons.
3. Inspect the [raw final outcomes](../results/v2/raw/main_rollouts.csv), [effective configurations](../configs/v2/main_plan.json) and [frozen protocol](../configs/v2/final_protocol.json).
4. Run the compact demo using the [reproduction guide](v2_reproduction.md).

The V2 snapshot improves instrumentation and reproducibility, without showing a nominal performance gain. The preserved V1 checkpoints remain the performance reference.

## What is public and what is local

The Git repository contains source, tests, configurations, reports, all final per-episode outcome tables, resource summaries and short videos. The small V2 EMA checkpoint bundle is a release asset with a [SHA-256 manifest](releases/v2_demo.json). It includes its recipe and raw expected demo traces.

The full HDF5 demonstration collection, training/resume checkpoints, caches and full 892,474-step evaluation traces remain under ignored local `artifacts/`. A clone plus the compact bundle can run the short demo; a clone alone cannot retrain the study or rerun the complete delivery audit. The [full local artifact map](artifact_publication.md) documents the V1 provenance. No full dataset release is claimed.

## Evidence map

| Question | Evidence |
| --- | --- |
| Did V2 improve manipulation? | [Fair before/after table](v2_report.md#fair-beforeafter), [raw V1 re-evaluation](../results/v2/raw/before_after_rollouts.csv) |
| Does fusion beat either single view? | [Summary and paired intervals](../results/v2/summary.json), including wrist-only |
| How much comes from the color prior? | Six matched point-policy variants in the [main plan](../configs/v2/main_plan.json) |
| Why does ACT fail? | [Train/inference diagnostics](../results/v2/diagnostics.json), [component interventions](../results/v2/act_component_interventions.json) |
| Is success stable after release? | [Metric](v2_design.md), [tests](../tests/v2/test_stability.py), [final trace audit](../results/v2/delivery_verification.json) |
| Can the checkpoint be resumed/replayed? | [Training verification](../results/v2/pretest_verification.json), [clean replay](../results/v2/clean_reproduction.json), [isolated replay](../results/v2/portable_demo_verification.json) |

Instruction conditioning uses two one-hot label pairs; the displayed sentence is generated from those labels. No learned text encoder is present. Teacher truth and evaluation geometry do not enter student observations. See [contracts](contracts.md), [architecture](architecture.md) and [V2 model cards](v2_model_cards.md).

## Continue the research

Follow [CONTRIBUTING](../CONTRIBUTING.md) and create a new experiment identity. Keep the published V2 metric and results fixed. Priority hypotheses remain selection/perception and learned motion control; a reliable ACT baseline and general instruction grounding have not been established.
