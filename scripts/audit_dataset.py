"""Audit actual episodes and optionally regenerate a portable dataset manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="artifacts/dataset")
p.add_argument("--out", default="artifacts/final_dataset_audit.json")
p.add_argument("--verify-frozen", action="store_true")
a = p.parse_args()
rows = []
teacher_sha = hashlib.sha256(Path("artifacts/teacher_selected.zip").read_bytes()).hexdigest()
for path in sorted(Path(a.dataset).glob("episode_*.h5")):
    with h5py.File(path) as f:
        row = json.loads(f.attrs["metadata"])
        seed = row["scene_seed"]
        n = row["frames"]
        assert row["split"] == (
            "train" if seed < 100000 else "validation" if seed < 200000 else "test"
        )
        assert row["split"] != "test", "Test must not appear in the demonstration dataset"
        assert row["provenance"] == "ppo_bc_initialized" and row["teacher_sha256"] == teacher_sha
        assert f["state"].shape == (n, 23) and f["action"].shape == (n, 7)
        assert np.isfinite(f["state"][:]).all() and np.isfinite(f["action"][:]).all()
        assert np.max(np.abs(f["action"][:])) <= 1.000001
        assert np.allclose(np.diff(f["timestamp_s"][:]), 0.05, rtol=0, atol=1e-8)
        tokens = np.r_[np.eye(2)[row["object_id"]], np.eye(2)[row["goal_id"]]].astype(np.float32)
        np.testing.assert_array_equal(f["instruction_tokens"][:], np.tile(tokens, (n, 1)))
        for camera in ["agentview", "robot0_eye_in_hand"]:
            assert (
                f[camera + "/rgb"].shape == (n, 128, 128, 3)
                and f[camera + "/rgb"].dtype == np.uint8
            )
            assert (
                f[camera + "/depth_m"].shape == (n, 128, 128)
                and f[camera + "/depth_m"].dtype == np.float32
            )
            assert f[camera + "/base_from_camera"].shape == (n, 4, 4)
        row["path"] = path.as_posix()
        row["bytes"] = path.stat().st_size
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
train = [r["episode_id"] for r in rows if r["split"] == "train" and r["success"]][:200]
validation = [r["episode_id"] for r in rows if r["split"] == "validation" and r["success"]]
assert len(train) == 200 and len({r["scene_seed"] for r in rows}) == len(rows)
report = {
    "episodes": rows,
    "selected_train_ids": train,
    "selected_validation_ids": validation,
    "raw_episodes": len(rows),
    "raw_bytes": sum(r["bytes"] for r in rows),
    "teacher_sha256": teacher_sha,
    "timestamp_action_sensor_contract_pass": True,
    "test_scenes_in_data": 0,
}
if a.verify_frozen:
    old = json.loads(Path("configs/dataset_manifest.json").read_text())
    assert old["selected_train_ids"] == train and old["selected_validation_ids"] == validation
    assert {r["episode_id"]: r["sha256"] for r in old["episodes"]} == {
        r["episode_id"]: r["sha256"] for r in rows
    }
    report["frozen_manifest_verified"] = True
Path(a.out).write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {k: v for k, v in report.items() if k != "episodes" and not k.startswith("selected_")},
        indent=2,
    )
)
