"""Reproducible dependency-only wheel adaptation for VLA+OSC simulation.

robosuite 1.5.2 requires old Mink/NumPy<2 although Panda OSC does not use
Mink. LeRobot requires NumPy>=2 via its visualization dependency. Preserve ALL
robosuite runtime/assets/license bytes; remove unused optional IK dependencies
from metadata in a clearly named local build. Do not use Mink controllers here.
"""

import base64
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

out = Path("artifacts/compatibility_wheels")
out.mkdir(parents=True, exist_ok=True)
release = json.load(urllib.request.urlopen("https://pypi.org/pypi/robosuite/1.5.2/json"))
entry = next(item for item in release["urls"] if item["filename"].endswith(".whl"))
upstream = out / entry["filename"]
if not upstream.exists():
    urllib.request.urlretrieve(entry["url"], upstream)
digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
assert digest == entry["digests"]["sha256"]
version = "1.5.2+geopolicyosc"
old_dist = "robosuite-1.5.2.dist-info/"
new_dist = f"robosuite-{version}.dist-info/"
target = out / f"robosuite-{version}-py3-none-any.whl"
records = []
removed = []
preserved = 0
with zipfile.ZipFile(upstream) as src, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as dst:
    for name in src.namelist():
        if name.endswith(".dist-info/RECORD"):
            continue
        payload = src.read(name)
        new_name = name.replace(old_dist, new_dist)
        if name.endswith(".dist-info/METADATA"):
            text = payload.decode()
            lines = []
            for line in text.splitlines():
                if line.startswith("Requires-Dist: mink") or line.startswith(
                    "Requires-Dist: qpsolvers"
                ):
                    removed.append(line)
                    continue
                lines.append(line.replace("Version: 1.5.2", "Version: " + version))
            payload = ("\n".join(lines) + "\n").encode()
        else:
            preserved += 1
        dst.writestr(new_name, payload)
        checksum = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        records.append([new_name, "sha256=" + checksum, len(payload)])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(records + [[new_dist + "RECORD", "", ""]])
    dst.writestr(new_dist + "RECORD", buffer.getvalue())
report = {
    "upstream": entry["url"],
    "upstream_sha256": digest,
    "local_version": version,
    "removed_optional_ik_requirements": removed,
    "preserved_runtime_asset_license_files": preserved,
    "target": str(target),
    "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "scope": "Panda OSC only. No simulator/controller/runtime code modification; NumPy2 compatibility must be tested.",
}
(out / "manifest.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
