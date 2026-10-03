"""Check GitHub-facing evidence using only repository files and the standard library.

Historical reports may link to excluded local artifacts. Those references are
counted explicitly; the public entry points must use repository or web links.
This check neither imports Torch nor requires checkpoints or demonstrations.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENTS = {"README.md", "docs/releases/v4.md"}
VERBATIM_SNAPSHOTS = {"README_original.md", "final_report_original.md"}


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(relative, normalize_newlines=False):
    content = (ROOT / relative).read_bytes()
    if normalize_newlines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def canonical_json_digest(relative):
    # V3 freezes canonical JSON values, not the file's bytes (unlike V4).
    content = json.dumps(
        read(relative), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(content).hexdigest()


def repository_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f"Path escapes repository: {relative}")
    return path


def frozen_text_matches(relative, expected):
    # Frozen Windows JSON files used byte hashes, while Git stores LF blobs.
    # Accept only that line-ending conversion in this repository-only check.
    # The release bundle and replay still require their original exact bytes.
    content = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    return expected in {
        hashlib.sha256(content).hexdigest(),
        hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest(),
    }


def check_links():
    checked = 0
    local_only = defaultdict(int)
    skipped = []
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"] + sorted(
        (ROOT / "docs").rglob("*.md")
    )
    require(
        all((ROOT / name).is_file() for name in PUBLIC_DOCUMENTS),
        "A public presentation document is missing",
    )
    for document in documents:
        relative = document.relative_to(ROOT).as_posix()
        # Preserve the original V1 snapshots and their original link bases.
        if document.name in VERBATIM_SNAPSHOTS:
            skipped.append(relative)
            continue
        content = document.read_text(encoding="utf8")
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        links += re.findall(
            r"<(?:img|video|source)\b[^>]*\bsrc=['\"]([^'\"]+)['\"]", content
        )
        for link in links:
            link = link.strip()
            if link.startswith("<") and ">" in link:
                link = link[1 : link.index(">")]
            else:
                # Support the optional Markdown title after a URL.
                link = re.split(r"\s+[\"']", link, maxsplit=1)[0]
            link = unquote(link.split("#", 1)[0].split("?", 1)[0])
            if not link or re.match(r"(?:https?://|mailto:)", link):
                continue
            target = (document.parent / link).resolve()
            require(
                target.is_relative_to(ROOT),
                f"Link escapes repository in {relative}: {link}",
            )
            target_relative = target.relative_to(ROOT).as_posix()
            if target_relative.split("/", 1)[0].lower() == "artifacts":
                require(
                    relative not in PUBLIC_DOCUMENTS,
                    f"Public document links to an excluded local artifact: {relative}: {link}",
                )
                local_only[relative] += 1
                continue
            require(target.exists(), f"Broken repository link in {relative}: {link}")
            checked += 1
    return dict(
        checked_repository_links=checked,
        historical_local_only_links=sum(local_only.values()),
        local_only_by_document=dict(local_only),
        verbatim_snapshots_skipped=skipped,
    )


def check_frozen_sources(protocol, field="sources"):
    for relative, expected in protocol[field].items():
        repository_path(relative)
        require(
            digest(relative, normalize_newlines=True) == expected,
            f"Frozen source changed: {relative}",
        )
    return len(protocol[field])


def check_v2():
    protocol = read("configs/v2/final_protocol.json")
    frozen = check_frozen_sources(protocol, "evaluation_source_hashes")
    summary = read("results/v2/summary.json")
    expected = {"main": 3150, "before_after": 300, "counterfactual": 360}
    require(
        summary["rollout_counts"] == expected, "Unexpected V2 final evaluation counts"
    )
    for group, count in expected.items():
        rows = read(f"results/v2/raw/{group}_rollouts.json")
        require(len(rows) == count, f"Wrong V2 raw outcome count: {group}")
    for item in read("docs/media/manifest.json")["animations"]:
        require(
            digest(item["path"]) == item["sha256"],
            f"V2 animation checksum mismatch: {item['path']}",
        )
    bundle = read("docs/releases/v2_demo.json")
    require(
        bundle["policy_sha256"]
        == read("configs/v2/demo_equivalence.json")["compact_sha256"],
        "V2 release checkpoint differs from verified compact checkpoint",
    )
    return dict(raw_outcomes=sum(expected.values()), frozen_sources=frozen)


def outcome_counts(rows, field):
    require(
        all(type(row[field]) is bool for row in rows), f"Non-boolean outcome: {field}"
    )
    return sum(row[field] for row in rows)


def check_v3():
    protocol = read("configs/v3/final_protocol.json")
    frozen = check_frozen_sources(protocol)
    require(
        canonical_json_digest("configs/v3/plan.json") == protocol["plan_sha256"],
        "V3 frozen plan changed",
    )
    summary = read("results/v3/final_summary.json")
    expected_scenes = set(
        range(protocol["first"], protocol["first"] + protocol["episodes"])
    )
    groups = defaultdict(list)
    expected_files = set()
    for record in protocol["registry"]:
        relative = f"results/v3/test/final_{record['key']}.json"
        expected_files.add(Path(relative).name)
        rows = read(relative)["rollouts"]
        require(len(rows) == protocol["episodes"], f"Wrong V3 row count: {relative}")
        require(
            {row["scene_seed"] for row in rows} == expected_scenes,
            f"Wrong V3 scenes: {relative}",
        )
        require(
            all(
                row["group"] == record["group"]
                and row["training_seed"] == record["seed"]
                and row["condition"] == "nominal"
                and row["checkpoint_sha256"] == record["checkpoint_sha256"]
                for row in rows
            ),
            f"V3 raw outcome identity differs from registry: {relative}",
        )
        groups[record["group"]].extend(rows)
    require(
        {p.name for p in (ROOT / "results/v3/test").glob("*.json")} == expected_files,
        "Unexpected or missing V3 test cell",
    )
    total = sum(len(rows) for rows in groups.values())
    require(total == summary["rollouts"] == 1500, "Unexpected V3 total")
    require(
        set(groups) == set(summary["groups"]),
        "V3 summary groups differ from raw outcomes",
    )
    for group, rows in groups.items():
        stats = summary["groups"][group]
        require(
            len(rows) == stats["episodes"], f"V3 summary row count differs: {group}"
        )
        for field, aggregate, per_seed in (
            ("physical_success", "physical", "per_seed_physical"),
            ("strict_v2_success", "strict", "per_seed_strict"),
        ):
            require(
                outcome_counts(rows, field) == stats[aggregate],
                f"V3 summary success count differs: {group}/{field}",
            )
            actual = [
                outcome_counts([r for r in rows if r["training_seed"] == seed], field)
                for seed in protocol["seeds"]
            ]
            require(
                actual == stats[per_seed],
                f"V3 summary seed counts differ: {group}/{field}",
            )
    require(
        summary["groups"]["v3_fusion_prior"]["physical"] == 147,
        "Unexpected V3 headline result",
    )
    return dict(raw_outcomes=total, frozen_sources=frozen)


def check_v4():
    protocol = read("configs/v4/final_protocol.json")
    frozen = check_frozen_sources(protocol)
    plan = read("configs/v4/plan.json")
    require(
        frozen_text_matches("configs/v4/plan.json", protocol["plan_sha256"]),
        "V4 frozen plan changed",
    )
    summary = read("results/v4/summary.json")
    require(
        summary["plan_sha256"] == protocol["plan_sha256"], "V4 summary plan differs"
    )
    require(
        frozen_text_matches(
            "configs/v4/final_protocol.json", summary["protocol_sha256"]
        ),
        "V4 summary protocol differs",
    )
    first, last = protocol["reserved_test_scenes"]
    expected_scenes = set(range(first, last + 1))
    grouped = defaultdict(list)
    expected_files = set()
    for record in protocol["registry"]:
        for condition in plan["conditions"]:
            relative = f"results/v4/test/final_{record['name']}_{condition}.json"
            expected_files.add(Path(relative).name)
            result = read(relative)
            rows = result["rollouts"]
            require(
                len(rows) == len(expected_scenes), f"Wrong V4 row count: {relative}"
            )
            require(
                {row["scene_seed"] for row in rows} == expected_scenes,
                f"Wrong V4 scenes: {relative}",
            )
            require(
                all(
                    row["group"] == record["group"]
                    and row["training_seed"] == record["training_seed"]
                    and row["condition"] == condition
                    and row["checkpoint_sha256"] == record["sha256"]
                    for row in rows
                ),
                f"V4 raw outcome identity differs from registry: {relative}",
            )
            identity = result["identity"]
            require(
                identity["sources"] == protocol["sources"],
                f"V4 raw source identity differs: {relative}",
            )
            require(
                identity["plan_sha256"] == protocol["plan_sha256"],
                f"V4 raw plan identity differs: {relative}",
            )
            grouped[record["group"], condition].extend(rows)
    require(
        {p.name for p in (ROOT / "results/v4/test").glob("*.json")} == expected_files,
        "Unexpected or missing V4 test cell",
    )
    total = sum(len(rows) for rows in grouped.values())
    require(total == summary["test"]["rollouts"] == 2400, "Unexpected V4 total")
    for (group, condition), rows in grouped.items():
        stats = summary["groups"][group][condition]
        require(
            len(rows) == stats["rollouts"],
            f"V4 summary row count differs: {group}/{condition}",
        )
        for field in ("physical_success", "strict_v2_success"):
            require(
                outcome_counts(rows, field) == stats[field]["successes"],
                f"V4 summary success count differs: {group}/{condition}/{field}",
            )
            actual = [
                outcome_counts([r for r in rows if r["training_seed"] == seed], field)
                for seed in summary["test"]["training_seeds"]
            ]
            require(
                actual == stats[field]["per_seed_successes"],
                f"V4 summary seed counts differ: {group}/{condition}/{field}",
            )
    for group, values in summary["groups"].items():
        stats = values["mean_perturbed"]
        rows = [
            row
            for condition in stats["conditions"]
            for row in grouped[group, condition]
        ]
        require(
            len(rows) == stats["rollouts"] == 540,
            f"Wrong V4 perturbed denominator: {group}",
        )
        for field in ("physical_success", "strict_v2_success"):
            require(
                outcome_counts(rows, field) == stats[field]["successes"],
                f"V4 perturbed mean count differs: {group}/{field}",
            )
    require(
        summary["groups"]["fusion_clean"]["mean_perturbed"]["physical_success"][
            "successes"
        ]
        == 483,
        "Unexpected V4 fusion headline result",
    )
    require(
        summary["groups"]["fixed_clean"]["mean_perturbed"]["physical_success"][
            "successes"
        ]
        == 272,
        "Unexpected V4 fixed headline result",
    )
    return dict(raw_outcomes=total, frozen_sources=frozen)


def check_asset_metadata(name, item):
    require(isinstance(item, dict), f"Invalid asset metadata: {name}")
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))),
        f"Invalid asset SHA-256: {name}",
    )
    require(
        type(item.get("size_bytes")) is int and item["size_bytes"] > 0,
        f"Invalid asset size: {name}",
    )


def check_v4_release():
    release = read("docs/releases/v4_demo.json")
    require(
        release["status"] == "prepared_locally_not_published",
        "V4 release status must describe the local preparation",
    )
    require(
        isinstance(release.get("release_tag"), str) and release["release_tag"],
        "Missing V4 release tag",
    )
    assets = release["assets"]
    require(isinstance(assets, dict) and bool(assets), "Missing V4 release assets")
    for name, item in assets.items():
        require(
            Path(name).name == name and name not in {".", ".."},
            f"Asset name must be a basename: {name}",
        )
        check_asset_metadata(name, item)
    protocol = read("configs/v4/final_protocol.json")
    exported = read("results/v4/demo_export.json")
    require(
        release["publication_url"] is None,
        "An unpublished release must not invent a download URL",
    )
    require(
        release["demo_policy"] == exported["name"] == "fusion_aug_s0"
        and release["demo_scene"] == exported["scene_seed"] == 500000
        and release["demo_condition"] == exported["condition"] == "absent"
        and release["result_policy"] == "fusion_clean",
        "The augmented demo and unaugmented aggregate must remain distinct",
    )
    require(
        release["contains_training_data"] is False
        and release["contains_optimizer"] is False,
        "The compact demo must be separate from retraining artifacts",
    )
    require(
        release["original_bundle_sha256"] == exported["bundle_sha256"],
        "V4 release original bundle identity differs",
    )
    require(
        release["checkpoint_sha256"] == exported["compact_sha256"],
        "V4 release compact checkpoint identity differs",
    )
    for name, sha_field, size_field in (
        ("geopolicy-v4-demo.zip", "bundle_sha256", "bundle_bytes"),
        ("checkpoint.pt", "compact_sha256", "compact_bytes"),
    ):
        require(
            assets[name]["sha256"] == exported[sha_field]
            and assets[name]["size_bytes"] == exported[size_field],
            f"V4 release asset differs from the original export: {name}",
        )
    members = release["members_sha256"]
    require(
        len(members) == release["archive_member_count"] == 79,
        "Wrong V4 bundle member count",
    )
    for relative, expected in members.items():
        repository_path(relative)
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", expected)),
            f"Malformed bundle member SHA-256: {relative}",
        )
    for name, field in (
        ("checkpoint.pt", "compact_sha256"),
        ("expected_initial.npz", "expected_initial_sha256"),
        ("expected_trace.json", "expected_trace_sha256"),
        ("expected_row.json", "expected_row_sha256"),
    ):
        require(
            members[f"artifacts/v4/demo/{name}"] == exported[field],
            f"V4 demo expected data differ: {name}",
        )
    for field in ("python_version", "packages"):
        require(
            release["runtime"][field] == protocol[field] == exported[field],
            f"V4 release runtime differs: {field}",
        )
    media = release["media"]
    require(
        isinstance(media, dict) and bool(media),
        "Missing V4 tracked demonstration media",
    )
    for relative, item in media.items():
        path = repository_path(relative)
        require(
            path.is_relative_to(ROOT / "docs/media"),
            f"V4 media must live in docs/media: {relative}",
        )
        check_asset_metadata(relative, item)
        require(
            path.is_file() and path.stat().st_size == item["size_bytes"],
            f"V4 tracked media size differs: {relative}",
        )
        require(
            digest(relative) == item["sha256"],
            f"V4 tracked media SHA-256 differs: {relative}",
        )
    # Asset binaries remain excluded from Git; their byte verification is the
    # separate local release check. CI verifies provenance and repository media.
    return dict(
        status=release["status"], asset_metadata=len(assets), tracked_media=len(media)
    )


def check():
    checks = dict(
        links=check_links(),
        v2=check_v2(),
        v3=check_v3(),
        v4=check_v4(),
        release_v4=check_v4_release(),
    )
    print(json.dumps(checks, indent=2))
    print(
        "Publication checks passed using repository files only; historical artifact links are local-only and V4 publication remains pending."
    )


if __name__ == "__main__":
    check()
