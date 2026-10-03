"""Summarize the recorded resource samples without launching a compute worker."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def aggregate(paths):
    samples = []
    for path in paths:
        samples.extend(json.loads(path.read_text()))
    assert samples
    return dict(
        sampled_jobs=len(paths),
        samples=len(samples),
        max_gpu_temperature_c=max(
            r["gpu_used_free_mib_temperature_c"][2] for r in samples
        ),
        max_gpu_used_mib=max(r["gpu_used_free_mib_temperature_c"][0] for r in samples),
        min_free_commit_gib=min(r["free_commit_bytes"] for r in samples) / 1024**3,
        max_worker_private_gib=max(r["private_bytes"] for r in samples) / 1024**3,
        min_free_disk_gib=min(r["free_disk_bytes"] for r in samples) / 1024**3,
    )


def main():
    protocol = json.loads((ROOT / "configs/v3/final_protocol.json").read_text())
    stages = json.loads(
        (ROOT / "artifacts/v3/pipeline/frozen_final_test/stages.json").read_text()
    )
    assert all(stages[r["key"]]["status"] == "complete" for r in protocol["registry"])
    evaluation = aggregate(
        [
            ROOT / f"artifacts/v3/evaluations/final_{r['key']}/resources.json"
            for r in protocol["registry"]
        ]
    )
    training = aggregate(sorted((ROOT / "artifacts/v3/runs").glob("*/resources.json")))
    result = dict(
        training=training,
        frozen_final_evaluation=evaluation,
        limits="Sampled process/system readings, not continuous peaks or power measurements. Training uses CUDA; policy evaluation uses CPU with two Torch threads and GPU rendering. Jobs are serialized.",
    )
    (ROOT / "results/v3/resource_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
