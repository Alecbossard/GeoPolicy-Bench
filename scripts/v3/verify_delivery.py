"""Verify the local delivered identities, raw rows, replay and Markdown links."""

import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def main():
    protocol = json.loads((ROOT / "configs/v3/final_protocol.json").read_text())
    summary = json.loads((ROOT / "results/v3/final_summary.json").read_text())
    assert summary["rollouts"] == 1500
    quality = json.loads(
        (ROOT / "results/v3/sensor_identity_quality_check.json").read_text()
    )
    assert summary["sensor_identity"]["matching_rollouts"] == 1499
    assert summary["sensor_identity"]["mismatches"] == quality["raw_mismatches"]
    assert all(
        row["actions_physics_exact"]
        for rows in quality["fresh_sequence_replays"].values()
        for row in rows
    )
    assert all(
        r["equal"] for r in quality["fresh_initial_modality_comparison"].values()
    )
    assert (
        summary["contrasts"]["v3_fusion_prior_minus_v3_original_bc"][
            "matched_scenes_per_seed"
        ]
        == 50
    )
    assert (
        summary["contrasts"]["v3_fusion_prior_minus_v3_fixed_prior"][
            "matched_scenes_per_seed"
        ]
        == 50
    )
    for name, expected in protocol["sources"].items():
        assert (
            hashlib.sha256(
                (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            == expected
        ), name
    assert (
        json_hash(json.loads((ROOT / "configs/v3/plan.json").read_text()))
        == protocol["plan_sha256"]
    )
    assert (
        json_hash(json.loads((ROOT / "configs/v3/runtime_limits.json").read_text()))
        == protocol["runtime_limits_sha256"]
    )
    count = 0
    for record in protocol["registry"]:
        assert sha(ROOT / record["checkpoint"]) == record["checkpoint_sha256"]
        saved = json.loads(
            (ROOT / f"results/v3/test/final_{record['key']}.json").read_text()
        )
        assert saved["identity"]["protocol_sha256"] == json_hash(protocol)
        rows = saved["rollouts"]
        assert len(rows) == 50 and [r["scene_seed"] for r in rows] == list(
            range(400000, 400050)
        )
        assert all(
            not r["diagnostic_oracle"] and not r["overfit_diagnostic"] for r in rows
        )
        with (ROOT / f"results/v3/test/final_{record['key']}.csv").open(
            newline="", encoding="utf8"
        ) as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 50
        for row, csv_row in zip(rows, csv_rows):
            assert all(str(value) == csv_row[key] for key, value in row.items())
        count += len(rows)
    with (ROOT / "results/v3/all_test_rollouts.csv").open(
        newline="", encoding="utf8"
    ) as stream:
        assert len(list(csv.DictReader(stream))) == count
    audited = json.loads((ROOT / "results/v3/trace_audit.json").read_text())
    assert all(
        f"final_{r['key']}" in audited["summaries"] for r in protocol["registry"]
    )
    datasets = 0
    for name in ("single_dataset.json", "two_objects_one_goal_dataset.json"):
        rows = json.loads((ROOT / "configs/v3" / name).read_text())
        assert len(rows) == 90
        for row in rows:
            assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
            datasets += 1
    proof = json.loads((ROOT / "configs/v3/demo_equivalence.json").read_text())
    assert sha(ROOT / "artifacts/v3/demo/checkpoint.pt") == proof["compact_sha256"]
    assert (
        sha(ROOT / "artifacts/v3/portable_demo/checkpoint.pt")
        == proof["compact_sha256"]
    )
    for name in (
        "artifacts/v3/demo/replay/verification.json",
        "artifacts/v3/portable_demo/replay/verification.json",
    ):
        replay = json.loads((ROOT / name).read_text())
        assert (
            replay["exact_frozen_trace"]
            and replay["physical_success"]
            and replay["strict_v2_success"]
        )
        assert replay["compact_sha256"] == proof["compact_sha256"]
        trace = json.loads((ROOT / name).with_name("trace.json").read_text())
        assert json_hash(trace) == proof["frozen_trace_sha256"]
    portable = ROOT / "artifacts/v3/portable_demo"
    replay = json.loads((portable / "replay/verification.json").read_text())
    assert (
        Path(replay["imported_source_directory"]).resolve()
        == (portable / "src/geopolicy").resolve()
    )
    for name, expected in protocol["sources"].items():
        assert (
            hashlib.sha256(
                (portable / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            == expected
        )
    bundle = json.loads((ROOT / "results/v3/demo_bundle.json").read_text())
    assert not bundle["published"] and sha(ROOT / bundle["path"]) == bundle["sha256"]
    with zipfile.ZipFile(ROOT / bundle["path"]) as archive:
        assert archive.testzip() is None
        assert (
            hashlib.sha256(archive.read("checkpoint.pt")).hexdigest()
            == proof["compact_sha256"]
        )
        for name, expected in protocol["sources"].items():
            assert (
                hashlib.sha256(archive.read(name).replace(b"\r\n", b"\n")).hexdigest()
                == expected
            )
        assert (
            json_hash(json.loads(archive.read("expected_trace.json")))
            == proof["frozen_trace_sha256"]
        )
    preserved = json.loads((ROOT / "results/v3/preservation_check.json").read_text())
    assert preserved["checked_files"] == 6210 and not preserved["failures"]
    links = 0
    for path in (ROOT / "docs/v3").glob("*.md"):
        for destination in re.findall(
            r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf8")
        ):
            if destination.startswith(("https://", "http://", "#")):
                continue
            target = destination.split("#", 1)[0].strip("<>")
            resolved = (path.parent / target).resolve()
            if resolved != ROOT / "results/v3/delivery_verification.json":
                assert resolved.exists(), f"Broken local link {path.name}: {target}"
            links += 1
    result = dict(
        verified_utc=datetime.now(timezone.utc).isoformat(),
        frozen_sources_exact=True,
        registered_checkpoints=30,
        raw_test_rows=count,
        csv_json_exact=True,
        initial_sensors_exact=False,
        matched_initial_sensor_rollouts=1499,
        sensor_identity_exception=summary["sensor_identity"]["mismatches"],
        sensor_quality_control_replay_episodes=30,
        sensor_quality_control_action_physics_traces_exact=True,
        dataset_files_sha_verified=datasets,
        compact_and_portable_replays_exact=True,
        historical_files_unchanged=6210,
        trace_audit_rows=audited["checked_rows"],
        trace_audit_steps=audited["checked_steps"],
        local_markdown_links=links,
    )
    (ROOT / "results/v3/delivery_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
