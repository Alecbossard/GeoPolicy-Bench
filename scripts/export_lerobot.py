"""Actual LeRobot dataset writer; lossless depth/points stay indexed in HDF5."""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/lerobot_dataset")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    manifest = json.loads(Path("configs/dataset_manifest.json").read_text())
    ids = manifest["selected_train_ids"] + manifest["selected_validation_ids"]
    if args.limit:
        ids = ids[: args.limit]
    by_id = {r["episode_id"]: r for r in manifest["episodes"]}
    features = {
        "observation.state": {"dtype": "float32", "shape": (23,), "names": None},
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        },
        "observation.images.fixed": {
            "dtype": "image",
            "shape": (3, 128, 128),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (3, 128, 128),
            "names": ["channels", "height", "width"],
        },
    }
    dataset = LeRobotDataset.create(
        "local/geopolicy-bench",
        fps=20,
        features=features,
        root=Path(args.out),
        robot_type="panda_osc_pose",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=2,
    )
    sidecars = []
    for episode_index, episode_id in enumerate(ids):
        row = by_id[episode_id]
        with h5py.File(row["path"]) as f:
            for frame_index in range(row["frames"]):
                dataset.add_frame(
                    {
                        "observation.state": f["state"][frame_index].astype(np.float32),
                        "action": f["action"][frame_index].astype(np.float32),
                        "observation.images.fixed": f["agentview/rgb"][frame_index],
                        "observation.images.wrist": f["robot0_eye_in_hand/rgb"][frame_index],
                        "task": row["instruction"],
                    }
                )
            dataset.save_episode()
        sidecars.append(
            {
                "lerobot_episode_index": episode_index,
                "source_episode_id": episode_id,
                "scene_seed": row["scene_seed"],
                "split": row["split"],
                "source_h5": row["path"],
                "source_sha256": row["sha256"],
                "frame_index_mapping": "identity",
                "depth_point_calibration_keys": [
                    "agentview",
                    "robot0_eye_in_hand",
                    "timestamp_s",
                    "world_from_base",
                ],
            }
        )
        print("Exported", episode_index, episode_id, row["frames"], flush=True)
    dataset.finalize()
    root = Path(args.out)
    (root / "depth_point_sidecars.json").write_text(json.dumps(sidecars, indent=2))
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    train_count = sum(by_id[i]["split"] == "train" for i in ids)
    info["splits"] = {"train": f"0:{train_count}"}
    if len(ids) > train_count:
        info["splits"]["validation"] = f"{train_count}:{len(ids)}"
    info_path.write_text(json.dumps(info, indent=2))
    # Reload the native dataset and verify episode/frame count and one real batch.
    reloaded = LeRobotDataset("local/geopolicy-bench", root=root, download_videos=False)
    sample = reloaded[0]
    with h5py.File(by_id[ids[0]]["path"]) as source:
        np.testing.assert_allclose(
            sample["observation.state"].numpy(), source["state"][0], rtol=0, atol=0
        )
        np.testing.assert_allclose(sample["action"].numpy(), source["action"][0], rtol=0, atol=0)
        for key, camera in [("fixed", "agentview"), ("wrist", "robot0_eye_in_hand")]:
            actual = (
                (sample[f"observation.images.{key}"].numpy().transpose(1, 2, 0) * 255)
                .round()
                .astype(np.uint8)
            )
            np.testing.assert_array_equal(actual, source[f"{camera}/rgb"][0])
    report = {
        "episodes": reloaded.num_episodes,
        "frames": len(reloaded),
        "features": list(sample),
        "state_shape": list(sample["observation.state"].shape),
        "action_shape": list(sample["action"].shape),
        "fixed_image_shape": list(sample["observation.images.fixed"].shape),
        "source_episode_ids": ids,
        "sidecars": str(root / "depth_point_sidecars.json"),
        "published": False,
        "reload_source_state_action_rgb_exact": True,
    }
    assert reloaded.num_episodes == len(ids) and len(reloaded) == sum(
        by_id[i]["frames"] for i in ids
    )
    (root / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
