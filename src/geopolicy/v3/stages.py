"""Progressive task complexity; physical scenes and manifests remain separate."""

import json
import os
import h5py
import numpy as np
import torch
from geopolicy.environment import SelectPlace, CAMERAS
from robosuite.environments.manipulation.lift import Lift
from geopolicy.sensors import camera_packet, fuse_points, student_state
from geopolicy.v2.data import history_state, view_names
from geopolicy.v2.resources import ResourceWatch
from .common import ROOT, sha, write
from .environment import SinglePlace
from .metrics import Tracker
from .data import Data


class TwoObjectsOneGoal(SinglePlace):
    def _load_model(self):
        SelectPlace._load_model(self)
        for body in list(self.model.worldbody):
            if body.get("name") in ("receptacle_1", "distractor"):
                self.model.worldbody.remove(body)

    def _setup_references(self):
        Lift._setup_references(self)
        self.object_ids = [
            self.cube_body_id,
            self.sim.model.body_name2id(self.other.root_body),
        ]
        self.goal_body_ids = [self.sim.model.body_name2id("receptacle_0")]

    def _reset_internal(self):
        Lift._reset_internal(self)
        jitter = self.rng_scene.uniform([-0.045, -0.025], [0.045, 0.025], (2, 2))
        slots = self.rng_scene.permutation(2)
        for i, obj in enumerate((self.cube, self.other)):
            position = np.array([0.04, -0.08 if slots[i] == 0 else 0.08, 0.83])
            position[:2] += jitter[i]
            self.sim.data.set_joint_qpos(obj.joints[0], np.r_[position, [1, 0, 0, 0]])
        slot = int(self.rng_scene.integers(2))
        goal = np.array([-0.1, -0.18 if slot == 0 else 0.18, 0.825])
        goal[:2] += self.rng_scene.uniform([-0.025, -0.014], [0.025, 0.014])
        self.goal_positions[0] = goal
        self.sim.model.body_pos[self.goal_body_ids[0]] = [goal[0], goal[1], 0.8]
        self.sim.forward()
        self.phase = self.phase_ticks = 0
        self.max_lift = 0.0
        self.collision_steps = 0
        self.wrong_target_any = self.wrong_destination_any = self.dropped_any = False

    def reset_scene(self, seed, object_id=None, goal_id=None):
        assert goal_id in (None, 0)
        return SelectPlace.reset_scene(self, seed, object_id, 0)

    def metrics(self):
        position = self.object_positions()[self.target_object]
        goal = self.goal_positions[0]
        obj = self.cube if self.target_object == 0 else self.other
        error = float(np.linalg.norm(position[:2] - goal[:2]))
        grasp = self._check_grasp(self.robots[0].gripper, obj)
        return dict(
            success=bool(
                error < 0.043
                and 0.816 < position[2] < 0.855
                and not grasp
                and self.max_lift > 0.89
            ),
            placement_error_m=error,
            collision=self.collision_steps > 0,
            collision_steps=self.collision_steps,
            max_lift_m=float(self.max_lift),
        )


def environment(task):
    return {
        "single": SinglePlace,
        "two_objects_one_goal": TwoObjectsOneGoal,
        "full": SelectPlace,
    }[task](cameras=True)


class StageTracker(Tracker):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.flags["wrong_object_lifted"] = False

    def update(self, env, obs, action, info):
        success = super().update(env, obs, action, info)
        wrong = bool(env.object_positions()[1 - env.target_object, 2] > 0.89)
        self.flags["wrong_object_lifted"] |= wrong
        self.trace[-1]["wrong_object_lifted"] = wrong
        self.trace[-1]["instruction_tokens"] = np.r_[
            np.eye(2)[env.target_object], np.eye(2)[env.target_goal]
        ].tolist()
        return success

    def summary(self):
        result = super().summary()
        if (
            not result["physical_success"]
            and self.flags["wrong_object_lifted"]
            and not self.flags["lifted"]
        ):
            result["failure_stage"] = "selection"
        return result


def collect_stage(task, split):
    assert task in ("two_objects_one_goal", "full") and split in ("train", "validation")
    stage = json.loads((ROOT / "configs/v3/task_stages.json").read_text())[task]
    first, episodes = stage[split + "_first"], stage[split + "_episodes"]
    plan = json.loads((ROOT / "configs/v3/plan.json").read_text())
    out = ROOT / "artifacts/v3" / f"{task}_dataset"
    out.mkdir(parents=True, exist_ok=True)
    env = environment(task)
    env.horizon = plan["maximum_steps"]
    watch = ResourceWatch(out, plan["resource_limits"])
    try:
        for seed in range(first, first + episodes):
            path = out / f"episode_{seed:06d}.h5"
            if path.exists():
                continue
            obs = env.reset_scene(seed)
            tracker = Tracker(plan["stability"])
            frames, original_frames = [], None
            for step in range(env.horizon):
                watch.sample()
                action = env.reference_action()
                packet = camera_packet(env, obs)
                packet.update(
                    state=student_state(obs).copy(),
                    action=action.copy(),
                    instruction_tokens=np.r_[
                        np.eye(2)[env.target_object], np.eye(2)[env.target_goal]
                    ].astype(np.float32),
                )
                frames.append(packet)
                obs, _, done, info = env.step(action)
                tracker.update(env, obs, action, info)
                if info["success"] and original_frames is None:
                    original_frames = len(frames)
                if done or (
                    original_frames is not None and len(frames) >= original_frames + 30
                ):
                    break
            metadata = dict(
                scene_seed=seed,
                episode_id=seed,
                split=split,
                success=bool(tracker.physical.success),
                original_frames=original_frames,
                frames=len(frames),
                instruction=env.instruction,
                task=task,
                object_id=env.target_object,
                goal_id=env.target_goal,
                provenance="scripted_ground_truth_phase_reference_DIAGNOSTIC_teacher",
                **tracker.summary(),
            )
            tmp = path.with_suffix(".tmp")
            with h5py.File(tmp, "w") as f:
                f.attrs["metadata"] = json.dumps(metadata)
                for key in (
                    "state",
                    "action",
                    "instruction_tokens",
                    "timestamp_s",
                    "world_from_base",
                ):
                    f.create_dataset(
                        key,
                        data=np.array([p[key] for p in frames]),
                        compression="gzip",
                        shuffle=True,
                    )
                for camera in CAMERAS:
                    group = f.create_group(camera)
                    for key in frames[0][camera]:
                        group.create_dataset(
                            key,
                            data=np.array([p[camera][key] for p in frames]),
                            compression="gzip",
                            shuffle=True,
                        )
            write(out / "teacher_traces" / f"{seed}.json", tracker.trace)
            os.replace(tmp, path)
            print(json.dumps(metadata), flush=True)
    finally:
        env.close()
    records = []
    for path in sorted(out.glob("*.h5")):
        with h5py.File(path) as f:
            records.append(
                dict(
                    json.loads(f.attrs["metadata"]),
                    path=path.relative_to(ROOT).as_posix(),
                    sha256=sha(path),
                )
            )
    pointer = ROOT / "configs/v3" / f"{task}_dataset.json"
    write(pointer, records)
    write(ROOT / "configs/v3/datasets" / f"{sha(pointer)}.json", records)


class StageData(Data):
    def __init__(self, cfg, split):
        self.cfg, self.horizon = cfg, cfg["horizon"]
        manifest = (
            ROOT / "configs/v3/datasets" / f"{cfg['dataset_manifest_sha256']}.json"
        )
        records = [
            r
            for r in json.loads(manifest.read_text())
            if r["split"] == split and r["success"]
        ]
        if cfg.get("limit"):
            records = records[: cfg["limit"]]
        self.episodes, self.index, self.identities = [], [], []
        cameras = view_names(cfg.get("view", "fusion"))
        for record in records:
            path = ROOT / record["path"]
            assert sha(path) == record["sha256"]
            count = record["frames"] if cfg["continued"] else record["original_frames"]
            assert count is not None
            with h5py.File(path) as f:
                item = {
                    k: f[k][:count] for k in ("state", "action", "instruction_tokens")
                }
                cached = {
                    c: (f[c]["points"][:count], f[c]["mask"][:count]) for c in cameras
                }
                points, masks = zip(
                    *(
                        fuse_points(
                            [(cached[c][0][t], cached[c][1][t]) for c in cameras]
                        )
                        for t in range(count)
                    )
                )
                item.update(points=np.stack(points), mask=np.stack(masks))
            eid = len(self.episodes)
            self.episodes.append(item)
            self.index.extend((eid, t) for t in range(count))
            self.identities.append(
                dict(path=record["path"], sha256=record["sha256"], frames=count)
            )


def stage_live(env, obs, history, cfg, norm):
    packet = camera_packet(env, obs)
    points, mask = fuse_points(
        [
            (packet[c]["points"], packet[c]["mask"])
            for c in view_names(cfg.get("view", "fusion"))
        ]
    )
    states = np.stack(history[-cfg["history"] :])
    batch = dict(
        state=history_state(states, len(states) - 1, cfg["history"], norm),
        instruction=np.r_[
            np.eye(2)[env.target_object], np.eye(2)[env.target_goal]
        ].astype(np.float32),
        points=points,
        point_mask=mask,
    )
    batch = {k: torch.from_numpy(np.asarray(v)[None]) for k, v in batch.items()}
    batch.update(
        action_mean=torch.from_numpy(norm["action_mean"]),
        action_std=torch.from_numpy(norm["action_std"]),
    )
    return batch
