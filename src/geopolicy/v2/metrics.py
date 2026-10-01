"""Evaluator-only stage diagnostics and sustained, released placement.

Object truth is used here to score behavior, never to construct student inputs.
The historical task and its instantaneous metric are unchanged.
"""

import itertools
import numpy as np


class StableWindow:
    def __init__(self, minimum_seconds):
        self.minimum_seconds = minimum_seconds
        self.started = None
        self.maximum_seconds = 0.0
        self.success = False

    def update(self, timestamp, valid):
        if not valid:
            self.started = None
            return False
        if self.started is None:
            self.started = timestamp
        elapsed = timestamp - self.started
        self.maximum_seconds = max(self.maximum_seconds, elapsed)
        self.success |= elapsed + 1e-9 >= self.minimum_seconds
        return self.success


def placement_valid(
    *,
    inside,
    height_valid,
    previously_lifted,
    finger_width,
    finger_contact,
    linear_speed,
    angular_speed,
    config,
):
    return bool(
        inside
        and height_valid
        and previously_lifted
        and not finger_contact
        and finger_width >= config["minimum_finger_width_m"]
        and linear_speed <= config["maximum_linear_speed_m_s"]
        and angular_speed <= config["maximum_angular_speed_rad_s"]
    )


class StageTracker:
    def __init__(self, config):
        self.config = config
        self.window = StableWindow(config["minimum_seconds"])
        self.trace = []
        self.flags = {
            key: False
            for key in [
                "target_approached",
                "selected_grasped",
                "selected_lifted",
                "wrong_object_lifted",
                "transport_reached",
                "released_in_tray",
                "instantaneous_success_v1",
            ]
        }

    def update(self, env, observation, action, info):
        cfg = self.config
        positions = env.object_positions()
        selected = env.target_object
        obj = [env.cube, env.other][selected]
        gripper = env.robots[0].gripper["right"]
        position = positions[selected]
        goal = env.goal_positions[env.target_goal]
        matrix = env.sim.data.body_xmat[env.object_ids[selected]].reshape(3, 3)
        corners = np.array(list(itertools.product([-1, 1], repeat=3))) * cfg["cube_half_extent_m"]
        corners = corners @ matrix.T + position
        offsets = np.abs(corners[:, :2] - goal[:2])
        bound = np.array(cfg["tray_inner_half_xy_m"]) - cfg["containment_margin_m"]
        inside = bool(np.all(offsets <= bound))
        qvel = np.asarray(env.sim.data.get_joint_qvel(obj.joints[0]))
        linear = float(np.linalg.norm(qvel[:3]))
        angular = float(np.linalg.norm(qvel[3:]))
        fingers = gripper.important_geoms["left_finger"] + gripper.important_geoms["right_finger"]
        finger_contact = bool(env.check_contact(fingers, obj.contact_geoms))
        width = float(np.abs(observation["robot0_gripper_qpos"]).sum())
        grasp = bool(env._check_grasp(gripper, obj))
        eef = observation["robot0_eef_pos"]
        distance = float(np.linalg.norm(eef - position))
        height_valid = cfg["center_height_m"][0] < position[2] < cfg["center_height_m"][1]
        lifted = env.max_lift > cfg["minimum_previous_lift_m"]
        released = bool(
            inside
            and height_valid
            and lifted
            and not finger_contact
            and width >= cfg["minimum_finger_width_m"]
        )
        self.flags["target_approached"] |= distance < 0.07
        self.flags["selected_grasped"] |= grasp
        self.flags["selected_lifted"] |= lifted
        self.flags["wrong_object_lifted"] |= (
            positions[1 - selected, 2] > cfg["minimum_previous_lift_m"]
        )
        self.flags["transport_reached"] |= bool(
            lifted and np.linalg.norm(position[:2] - goal[:2]) < 0.05
        )
        self.flags["released_in_tray"] |= released
        self.flags["instantaneous_success_v1"] |= info["success"]
        valid = placement_valid(
            inside=inside,
            height_valid=height_valid,
            previously_lifted=lifted,
            finger_width=width,
            finger_contact=finger_contact,
            linear_speed=linear,
            angular_speed=angular,
            config=cfg,
        )
        self.window.update(float(env.sim.data.time), valid)
        self.trace.append(
            {
                "time_s": float(env.sim.data.time),
                "gripper_command": float(action[6]),
                "finger_width_m": width,
                "finger_contact_selected": finger_contact,
                "grasp_selected": grasp,
                "eef_selected_distance_m": distance,
                "eef_other_distance_m": float(np.linalg.norm(eef - positions[1 - selected])),
                "selected_xyz_m": position.tolist(),
                "selected_goal_xy_error_m": float(np.linalg.norm(position[:2] - goal[:2])),
                "selected_linear_speed_m_s": linear,
                "selected_angular_speed_rad_s": angular,
                "full_xy_containment": inside,
                "valid_stability_sample": valid,
                "stable_success": self.window.success,
            }
        )
        return self.window.success

    def summary(self):
        f = self.flags
        if self.window.success:
            stage = "success"
        elif f["wrong_object_lifted"] and not f["selected_lifted"]:
            stage = "selection"
        elif not f["target_approached"]:
            stage = "selection_no_approach"
        elif not f["selected_lifted"]:
            stage = "grasp"
        elif not f["transport_reached"]:
            stage = "transport"
        elif not f["released_in_tray"]:
            stage = "release"
        else:
            stage = "unstable_placement"
        return dict(
            {key: bool(value) for key, value in f.items()},
            stable_success=self.window.success,
            failure_stage=stage,
            longest_stable_seconds=self.window.maximum_seconds,
        )
