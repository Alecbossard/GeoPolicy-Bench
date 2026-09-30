"""Verify portable final source fingerprints separately from the original freeze."""

import argparse
import hashlib
import json
from pathlib import Path

from geopolicy.io import save_json

p = argparse.ArgumentParser()
p.add_argument(
    "--write",
    action="store_true",
    help="Record final reviewed sources; never edits frozen protocol",
)
a = p.parse_args()
path = Path("configs/final_execution_sources.json")
files = sorted(file for root in ["src", "scripts", "tests"] for file in Path(root).rglob("*.py"))
actual = {
    file.as_posix(): hashlib.sha256(file.read_text(encoding="utf8").encode("utf8")).hexdigest()
    for file in files
}
manifest = json.loads(Path("configs/dataset_manifest.json").read_text(encoding="utf8"))
dataset_sha = hashlib.sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if a.write:
    save_json(
        path,
        {
            "note": "Final reviewed source fingerprints, with universal newline normalization for portable Git checkouts. The original frozen protocol retains original fingerprints. Post-freeze changes preserve primary recipe: exact-equality data I/O optimization, inference-only export, row resume/atomic writes, counterfactual hashes, guards, video endpoints, formatting and predeclared secondary dropout support. Trained primary weights remain unchanged.",
            "source_sha256_normalized_newlines": actual,
            "dataset_manifest_sha256_canonical_json": dataset_sha,
            "analysis_plan": "configs/analysis_plan.json",
        },
    )
    print(f"Recorded {len(actual)} source fingerprints")
else:
    expected = json.loads(path.read_text(encoding="utf8"))
    assert (
        actual == expected["source_sha256_normalized_newlines"]
    ), "Reviewed execution sources changed"
    assert (
        dataset_sha == expected["dataset_manifest_sha256_canonical_json"]
    ), "Frozen dataset manifest changed"
    print(f"Verified {len(actual)} source fingerprints and canonical dataset manifest")
