import json
import h5py
import numpy as np
from geopolicy.data import Episodes
from geopolicy.sensors import fuse_points
from geopolicy.environment import CAMERAS


def test_cached_points_and_dropout_preserve_sensor_only_contract(tmp_path):
    rng = np.random.default_rng(9)
    arrays = []
    with h5py.File(tmp_path / "episode.h5", "w") as f:
        f.attrs["metadata"] = json.dumps({"split": "train", "success": True, "teacher_phase": 7})
        f["state"] = rng.normal(size=(3, 23)).astype(np.float32)
        f["action"] = rng.normal(size=(3, 7)).astype(np.float32)
        f["instruction_tokens"] = np.tile([1, 0, 0, 1], (3, 1)).astype(np.float32)
        for j, c in enumerate(CAMERAS):
            points = rng.random((3, 32, 6), dtype=np.float32)
            points[:, :, 0] += j * 2
            mask = np.ones((3, 32), bool)
            f[c + "/points"] = points
            f[c + "/mask"] = mask
            arrays.append((points, mask))
    full = Episodes(tmp_path, limit=1)
    for t in range(3):
        p, m = fuse_points([(x[t], y[t]) for x, y in arrays])
        np.testing.assert_array_equal(full.episodes[0]["points"][t], p)
        np.testing.assert_array_equal(full.episodes[0]["mask"][t], m)
    dropout = Episodes(tmp_path, limit=1, view_dropout=1)
    batch = dropout.batch(np.random.default_rng(0), 12, dropout.normalization())
    baseline = full.batch(np.random.default_rng(0), 12, full.normalization())
    for key in ["state", "instruction", "action", "action_mask"]:
        np.testing.assert_array_equal(batch[key].numpy(), baseline[key].numpy())
    assert set(batch) == {"state", "instruction", "action", "action_mask", "points", "point_mask"}
    for points, mask in zip(batch["points"].numpy(), batch["point_mask"].numpy()):
        x = points[mask, 0]
        assert np.all(x < 1) or np.all(x > 2)
