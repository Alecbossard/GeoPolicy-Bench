"""Physical post-release dwell and unchanged V2 composite, evaluator only."""

import numpy as np
from geopolicy.v2.metrics import StableWindow, cube_inside_tray, placement_valid


class Tracker:
    def __init__(self, config):
        self.config = config
        self.physical = StableWindow(config["minimum_seconds"])
        self.strict = StableWindow(config["minimum_seconds"])
        self.release_seen = False
        self.flags = dict(
            approached=False, grasped=False, lifted=False, transported=False
        )
        self.trace = []

    def update(self, env, obs, action, info):
        cfg = self.config
        selected = env.target_object
        obj = env.cube if selected == 0 else env.other
        position = env.object_positions()[selected].copy()
        goal = env.goal_positions[env.target_goal].copy()
        rotation = env.sim.data.body_xmat[env.object_ids[selected]].reshape(3, 3).copy()
        velocity = np.asarray(env.sim.data.get_joint_qvel(obj.joints[0])).copy()
        linear, angular = float(np.linalg.norm(velocity[:3])), float(
            np.linalg.norm(velocity[3:])
        )
        gripper = env.robots[0].gripper["right"]
        fingers = (
            gripper.important_geoms["left_finger"]
            + gripper.important_geoms["right_finger"]
        )
        contact = bool(env.check_contact(fingers, obj.contact_geoms))
        width = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        inside = cube_inside_tray(position, rotation, goal, cfg)
        height = cfg["center_height_m"][0] < position[2] < cfg["center_height_m"][1]
        lifted = bool(env.max_lift > cfg["minimum_previous_lift_m"])
        self.release_seen |= bool(
            inside
            and height
            and lifted
            and not contact
            and width >= cfg["minimum_finger_width_m"]
        )
        geometry = dict(
            inside=inside,
            height_valid=height,
            previously_lifted=lifted,
            finger_width=width,
            finger_contact=contact,
            linear_speed=linear,
            angular_speed=angular,
            config=cfg,
        )
        strict_valid = placement_valid(**geometry)
        physical_valid = physical_sample(
            self.release_seen, inside, height, contact, linear, angular, cfg
        )
        timestamp = float(env.sim.data.time)
        self.physical.update(timestamp, physical_valid)
        self.strict.update(timestamp, strict_valid)
        distance = float(np.linalg.norm(obs["robot0_eef_pos"] - position))
        self.flags["approached"] |= distance < 0.07
        self.flags["grasped"] |= bool(env._check_grasp(gripper, obj))
        self.flags["lifted"] |= lifted
        self.flags["transported"] |= (
            lifted and np.linalg.norm(position[:2] - goal[:2]) < 0.05
        )
        self.trace.append(
            dict(
                time_s=timestamp,
                action=np.asarray(action).tolist(),
                selected_xyz_m=position.tolist(),
                rotation=rotation.tolist(),
                goal_xyz_m=goal.tolist(),
                eef_selected_distance_m=distance,
                linear_speed_m_s=linear,
                angular_speed_rad_s=angular,
                finger_width_m=width,
                finger_contact=contact,
                inside=inside,
                previously_lifted=lifted,
                release_seen=self.release_seen,
                physical_valid=physical_valid,
                strict_valid=strict_valid,
                physical_success=self.physical.success,
                strict_v2_success=self.strict.success,
                instantaneous_v1_success=bool(info["success"]),
            )
        )
        return self.physical.success and self.strict.success

    def summary(self):
        if self.physical.success:
            stage = "success"
        elif not self.flags["approached"]:
            stage = "approach"
        elif not self.flags["lifted"]:
            stage = "grasp"
        elif not self.flags["transported"]:
            stage = "transport"
        elif not self.release_seen:
            stage = "release"
        else:
            stage = "object_stability"
        return dict(
            **{key: bool(value) for key, value in self.flags.items()},
            release_seen=self.release_seen,
            physical_success=self.physical.success,
            strict_v2_success=self.strict.success,
            posture_only_failure=self.physical.success and not self.strict.success,
            failure_stage=stage,
            physical_dwell_s=self.physical.maximum_seconds,
            strict_dwell_s=self.strict.maximum_seconds,
        )


def physical_sample(released, inside, height, contact, linear, angular, cfg):
    return bool(
        released
        and inside
        and height
        and not contact
        and linear <= cfg["maximum_linear_speed_m_s"]
        and angular <= cfg["maximum_angular_speed_rad_s"]
    )
