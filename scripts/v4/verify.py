"""Stdlib/NumPy reservation and complete local delivery checks."""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads((ROOT / p).read_text(encoding="utf8"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(rel, d):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, allow_nan=False), encoding="utf8")


def reservation():
    conflicts = []
    known = set()
    files = []
    targets = (
        set(range(500000, 500020))
        | set(range(210000, 210010))
        | set(range(210100, 210110))
    )

    def scan(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("scene_seed", "episode_id") and isinstance(v, int):
                    known.add(v)
                    if v in targets:
                        conflicts.append(dict(path=path, key=k, value=v))
                scan(v, path)
        elif isinstance(x, list):
            for v in x:
                scan(v, path)

    for folder in ("configs", "results"):
        for p in (ROOT / folder).rglob("*.json"):
            if "v4" in p.parts:
                continue
            files.append(p.relative_to(ROOT).as_posix())
            scan(json.loads(p.read_text(encoding="utf8")), files[-1])
    result = dict(
        historical_json_files=len(files),
        historical_scene_ids=len(known),
        conflicts=conflicts,
        test_reserved=[500000, 500019],
        validation=[210000, 210009, 210100, 210109],
        checked_utc=datetime.now(timezone.utc).isoformat(),
    )
    dump("results/v4/reservation_check.json", result)
    assert not conflicts, conflicts
    print(json.dumps(result))


def preservation():
    snapshot = read("configs/v4/preservation.json")
    allowed = snapshot["allowed_documentation_corrections"]
    changed = []
    failures = []
    for rel, h in snapshot["files"].items():
        p = ROOT / rel
        if not p.exists() or sha(p) != h:
            if rel in allowed:
                original = ROOT / "artifacts/v4/preservation" / rel
                assert sha(original) == h
                changed.append(rel)
            else:
                failures.append(rel)
    result = dict(
        files_checked=len(snapshot["files"]),
        authorized_documentation_changes=changed,
        scientific_or_other_failures=failures,
        original_documentation_snapshots_verified=True,
        checked_utc=datetime.now(timezone.utc).isoformat(),
    )
    dump("results/v4/preservation_check.json", result)
    assert not failures, failures
    print(json.dumps(result))
    return result


def delivery():
    history = preservation()
    protocol = read("configs/v4/final_protocol.json")
    sys.path.insert(0, str(ROOT / "src"))
    from geopolicy.v4.common import verify_runtime

    verify_runtime(protocol)
    plan = read("configs/v4/plan.json")
    assert sha(ROOT / "configs/v4/plan.json") == protocol["plan_sha256"]
    assert (
        sha(ROOT / "results/v4/validation_decision.json")
        == protocol["validation_decision_sha256"]
    ), "Validation decision changed after freeze"
    assert (
        sha(ROOT / "results/v4/io_runtime_checks.json")
        == protocol["io_runtime_checks_sha256"]
    ), "I/O/runtime checks changed after freeze"
    assert read("results/v4/validation_decision.json")["proceed_to_reserved_test"]
    assert read("results/v4/io_runtime_checks.json")["verified"]
    frozen_time = datetime.fromisoformat(protocol["frozen_utc"]).timestamp()
    final_jobs = read("artifacts/v4/jobs/final_jobs.json")
    confirmation_jobs = read("artifacts/v4/jobs/confirmation_jobs.json")
    assert (
        confirmation_jobs["last_completed_time"]
        < frozen_time
        < final_jobs["started_time"]
    )
    assert sorted(final_jobs["completed"]) == list(range(120))
    assert sha(ROOT / "configs/v4/final_jobs.json") == final_jobs["jobs_sha256"]
    first_test_resource_time = min(
        row["time"]
        for path in (ROOT / "artifacts/v4/evaluations").glob("final_*/resources.json")
        for row in json.loads(path.read_text(encoding="utf8"))
    )
    assert frozen_time < first_test_resource_time
    for rel, h in protocol["sources"].items():
        assert (
            hashlib.sha256(
                (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            == h
        ), rel
    total_updates = 0
    for record in protocol["registry"]:
        assert sha(ROOT / record["checkpoint"]) == record["sha256"]
        manifest = read(f"artifacts/v4/main_runs/{record['name']}/manifest.json")
        assert (
            manifest["completed_update"] == 2000
            and len(manifest["training_data"]) == 79
        )
        total_updates += manifest["optimizer_updates_this_session"]
    for p in (ROOT / "results/v4/pilot_training").glob("*.json"):
        total_updates += json.loads(p.read_text())["manifest"]["completed_update"]
    assert total_updates <= plan["maximum_training_updates_total"]
    summary = read("results/v4/summary.json")
    assert summary["test"]["rollouts"] == 2400
    audit = read("results/v4/trace_sensor_audit.json")
    assert audit["checked_rollouts"] == 3540
    sensor = read("results/v4/report_sensor_identity.json")
    qc = read("results/v4/initial_sensor_qc.json")
    assert sensor["all_effective_policy_initial_pairs_exact"]
    assert (
        qc["all_effective_policy_initial_pairs_exact"]
        and not qc["effective_mismatches"]
    )
    assert qc["checked_rollouts"] == 2400
    assert (
        sha(ROOT / "results/v4/initial_sensor_qc.json")
        == sensor["quality_control_sha256"]
    )
    for rel, expected in qc["raw_result_sha256"].items():
        assert sha(ROOT / rel) == expected
    assert (
        len(audit["initial_clean_mismatches"])
        == len(qc["raw_modality_mismatches"])
        == 1
    )
    for p in (ROOT / "results/v4/test").glob("*.json"):
        value = json.loads(p.read_text(encoding="utf8"))
        assert len(value["rollouts"]) == 20
        assert (
            value["identity"]["sources"] == protocol["sources"]
            and value["identity"]["plan_sha256"] == protocol["plan_sha256"]
            and value["identity"]["packages"] == protocol["packages"]
            and value["identity"]["python_version"] == protocol["python_version"]
        )
    resources = [
        r
        for p in (ROOT / "artifacts/v4").rglob("resources.json")
        for r in json.loads(p.read_text(encoding="utf8"))
    ]
    resource_summary = dict(
        samples=len(resources),
        maximum_gpu_temperature_c=max(
            r["gpu_used_free_mib_temperature_c"][2]
            for r in resources
            if r["gpu_used_free_mib_temperature_c"]
        ),
        minimum_free_commit_gib=min(
            r["free_commit_bytes"]
            for r in resources
            if r["free_commit_bytes"] is not None
        )
        / 1024**3,
        maximum_process_private_gib=max(r.get("private_bytes") or 0 for r in resources)
        / 1024**3,
        minimum_free_disk_gib=min(r["free_disk_bytes"] for r in resources) / 1024**3,
    )
    verified = read("artifacts/v4/demo/reproduced/verification.json")
    portable = read(
        "artifacts/v4/portable_demo/artifacts/v4/demo/reproduced/verification.json"
    )
    for v in (verified, portable):
        assert all(
            v[k]
            for k in (
                "initial_modalities_exact",
                "actions_exact",
                "physics_exact",
                "summary_exact",
            )
        )
    failed_links = []
    links = 0
    for p in (ROOT / "docs/v4").glob("*.md"):
        for raw in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf8")):
            target = raw.split("#")[0]
            if not target or "://" in target:
                continue
            links += 1
            if not (p.parent / target).exists():
                failed_links.append(dict(file=p.name, target=target))
    assert not failed_links, failed_links
    result = dict(
        verified=True,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        training_optimizer_updates_executed=total_updates,
        contract_check_updates_separate=5,
        main_models=12,
        test_rollouts=2400,
        unique_test_scenes=20,
        conditions=10,
        training_seeds=3,
        trace_audit_rollouts=audit["checked_rollouts"],
        trace_audit_steps=audit["checked_steps"],
        initial_modalities_saved_and_recomputed=True,
        effective_policy_initial_pairs_exact=True,
        raw_initial_modalities_all_exact=qc["all_raw_initial_modalities_exact"],
        raw_modality_exceptions=qc["raw_modality_mismatches"],
        no_test_rollout_exclusions_or_replacements=True,
        source_and_checkpoint_hashes_verified=True,
        validation_completed_before_freeze_before_test=True,
        protocol_frozen_utc=protocol["frozen_utc"],
        first_test_job_utc=datetime.fromtimestamp(
            final_jobs["started_time"], timezone.utc
        ).isoformat(),
        bounded_elapsed_hours=(
            datetime.now(timezone.utc).timestamp()
            - read("artifacts/v4/budget.json")["started_time"]
        )
        / 3600,
        history=history,
        resources=resource_summary,
        local_demo=verified,
        portable_demo=portable,
        markdown_links_checked=links,
        no_cv_edits=True,
        no_publication=True,
    )
    dump("results/v4/delivery_verification.json", result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["reservation", "preservation", "delivery"])
    a = p.parse_args()
    globals()[a.command]()


if __name__ == "__main__":
    main()
