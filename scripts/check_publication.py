"""Check repository-only evidence and local documentation links without GPU/artifacts."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf8"))


def check():
    checked_links = 0
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"] + list((ROOT / "docs").rglob("*.md"))
    for document in documents:
        # Verbatim V1 snapshots retain links relative to their original location.
        if document.name in {"README_original.md", "final_report_original.md"}:
            continue
        content = document.read_text(encoding="utf8")
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        links += re.findall(r'<img[^>]+src="([^"]+)"', content)
        for link in links:
            link = link.strip().split("#", 1)[0]
            if not link or re.match(r"(?:https?://|mailto:)", link):
                continue
            target = (document.parent / link).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                raise ValueError(f"Broken local link in {document.relative_to(ROOT)}: {link}")
            checked_links += 1
    protocol = read("configs/v2/final_protocol.json")
    for path, digest in protocol["evaluation_source_hashes"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != digest:
            raise ValueError(f"Frozen evaluator source changed: {path}")
    summary = read("results/v2/summary.json")
    expected = {"main": 3150, "before_after": 300, "counterfactual": 360}
    if summary["rollout_counts"] != expected:
        raise ValueError("Unexpected final evaluation counts")
    for group, count in expected.items():
        rows = read(f"results/v2/raw/{group}_rollouts.json")
        if len(rows) != count:
            raise ValueError(f"Raw final outcome table has wrong row count: {group}")
    media = read("docs/media/manifest.json")
    for item in media["animations"]:
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Animation checksum mismatch: {item['path']}")
    bundle = read("docs/releases/v2_demo.json")
    if bundle["policy_sha256"] != read("configs/v2/demo_equivalence.json")["compact_sha256"]:
        raise ValueError("Release checkpoint differs from verified compact checkpoint")
    print(f"Publication checks passed: {checked_links} local links, frozen sources, 3810 raw outcomes, demo identities.")


if __name__ == "__main__":
    check()
