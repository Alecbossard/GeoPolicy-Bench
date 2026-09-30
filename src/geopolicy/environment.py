"""Two-object/two-receptacle Panda task. Ground truth is teacher/evaluator only."""

import numpy as np
from .runtime import setup

setup()
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_body, new_geom
from robosuite.controllers import load_composite_controller_config

COLORS = ("red", "green")
GOALS = ("blue", "yellow")
CAMERAS = ("agentview", "robot0_eye_in_hand")


class SelectPlace(Lift):
    def __init__(self, *, cameras=False, resolution=128, seed=0, **kwargs):
        self.scene_seed = seed
        self.target_object = 0
        self.target_goal = 0
        self.phase = 0  # privileged teacher curriculum; never in student observations
        self.phase_ticks = 0
        self.goal_positions = np.array([[-0.10, -0.18, 0.825], [-0.10, 0.18, 0.825]])
        self.max_lift = 0.0
        self.collision_steps = 0
        self.wrong_target_any = False
        self.wrong_destination_any = False
        self.dropped_any = False
        self.rng_scene = np.random.default_rng(seed)
        controller = load_composite_controller_config(controller="BASIC")
        controller["body_parts"] = {"right": controller["body_parts"]["right"]}
        super().__init__(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=cameras,
            use_camera_obs=cameras,
            use_object_obs=False,
            hard_reset=False,
            camera_names=list(CAMERAS),
            camera_heights=resolution,
            camera_widths=resolution,
            camera_depths=True,
            seed=seed,
            initialization_noise=None,
            controller_configs=controller,
            control_freq=20,
            horizon=220,
            reward_shaping=True,
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()
        # Constant dimensions/color identity; reset positions vary by scene.
        for geom in self.model.worldbody.iter("geom"):
            if geom.get("name", "").startswith("cube_") and geom.get("type") == "box":
                geom.set("size", ".021 .021 .021")
        self.other = BoxObject(
            name="green_cube", size=[0.021] * 3, rgba=[0, 0.8, 0.1, 1], friction=[1, 0.005, 0.0001]
        )
        self.model.merge_objects([self.other])
        for j, p in enumerate(self.goal_positions):
            body = new_body(name=f"receptacle_{j}", pos=[p[0], p[1], 0.8])
            color = [0.05, 0.2, 1, 1] if j == 0 else [1, 0.85, 0.02, 1]
            body.append(
                new_geom(
                    name=f"goal_floor_{j}",
                    type="box",
                    size=[0.065, 0.06, 0.004],
                    pos=[0, 0, 0.004],
                    rgba=color,
                    group=1,
                )
            )
            for k, (pos, size) in enumerate(
                [
                    ([0.067, 0, 0.018], [0.004, 0.064, 0.018]),
                    ([-0.067, 0, 0.018], [0.004, 0.064, 0.018]),
                    ([0, 0.064, 0.018], [0.063, 0.004, 0.018]),
                    ([0, -0.064, 0.018], [0.063, 0.004, 0.018]),
                ]
            ):
                body.append(
                    new_geom(
                        name=f"goal_wall_{j}_{k}",
                        type="box",
                        size=size,
                        pos=pos,
                        rgba=color,
                        group=1,
                    )
                )
            self.model.worldbody.append(body)
        # Non-target distractor away from initial object sites.
        body = new_body(name="distractor", pos=[0.16, 0, 0.835])
        body.append(
            new_geom(
                name="distractor_geom",
                type="box",
                size=[0.02] * 3,
                rgba=[0.45, 0.45, 0.45, 1],
                group=1,
            )
        )
        self.model.worldbody.append(body)

    def _setup_references(self):
        super()._setup_references()
        self.object_ids = [self.cube_body_id, self.sim.model.body_name2id(self.other.root_body)]
        self.goal_body_ids = [self.sim.model.body_name2id(f"receptacle_{i}") for i in range(2)]

    def _reset_internal(self):
        super()._reset_internal()
        jitter = self.rng_scene.uniform([-0.045, -0.025], [0.045, 0.025], (2, 2))
        slots = self.rng_scene.permutation(2)
        for i, obj in enumerate([self.cube, self.other]):
            pos = np.array([0.04, -0.08 if slots[i] == 0 else 0.08, 0.83])
            pos[:2] += jitter[i]
            self.sim.data.set_joint_qpos(obj.joints[0], np.r_[pos, [1, 0, 0, 0]])
        goal_slots = self.rng_scene.permutation(2)
        for j, body_id in enumerate(self.goal_body_ids):
            p = np.array([-0.10, -0.18 if goal_slots[j] == 0 else 0.18, 0.825])
            p[:2] += self.rng_scene.uniform([-0.025, -0.014], [0.025, 0.014])
            self.goal_positions[j] = p
            self.sim.model.body_pos[body_id] = [p[0], p[1], 0.8]
        self.sim.forward()
        self.phase = self.phase_ticks = 0
        self.max_lift = 0.0
        self.collision_steps = 0
        self.wrong_target_any = self.wrong_destination_any = self.dropped_any = False

    def reset_scene(self, seed, object_id=None, goal_id=None):
        self.scene_seed = int(seed)
        self.rng_scene = np.random.default_rng(seed)
        self.target_object = int(seed % 2 if object_id is None else object_id)
        self.target_goal = int((seed // 2) % 2 if goal_id is None else goal_id)
        return self.reset()

    @property
    def instruction(self):
        return f"Place the {COLORS[self.target_object]} cube in the {GOALS[self.target_goal]} receptacle."

    def object_positions(self):
        return np.array([self.sim.data.body_xpos[i] for i in self.object_ids])

    def metrics(self):
        positions = self.object_positions()
        goal = self.goal_positions[self.target_goal]
        p = positions[self.target_object]
        error = np.linalg.norm(p[:2] - goal[:2])
        # Released, settled inside correct tray, selected object actually lifted.
        obj = [self.cube, self.other][self.target_object]
        grasp = self._check_grasp(self.robots[0].gripper, obj)
        success = bool(
            error < 0.043 and 0.816 < p[2] < 0.855 and not grasp and self.max_lift > 0.89
        )
        wrong = any(
            np.linalg.norm(positions[1 - self.target_object, :2] - g[:2]) < 0.043
            for g in self.goal_positions
        )
        wrong_goal = self.goal_positions[1 - self.target_goal]
        wrong_destination = (
            np.linalg.norm(p[:2] - wrong_goal[:2]) < 0.043 and 0.816 < p[2] < 0.855 and not grasp
        )
        return {
            "success": success,
            "wrong_target": bool(wrong or self.wrong_target_any),
            "wrong_destination": bool(wrong_destination or self.wrong_destination_any),
            "placement_error_m": float(error),
            "placement_error_xyz_m": float(np.linalg.norm(p - goal)),
            "dropped": bool((positions[:, 2] < 0.75).any() or self.dropped_any),
            "max_lift_m": self.max_lift,
            "collision": self.collision_steps > 0,
            "collision_steps": self.collision_steps,
        }

    def forbidden_contact(self):
        for contact in self.sim.data.contact:
            if contact.dist > -0.001:
                continue
            names = [
                self.sim.model.geom_id2name(int(i)) or "" for i in [contact.geom1, contact.geom2]
            ]
            for i in range(2):
                robot = names[i].startswith(("robot0_", "gripper0_"))
                obstacle = (
                    "table_collision" in names[1 - i]
                    or "goal_wall" in names[1 - i]
                    or "distractor_geom" in names[1 - i]
                )
                if robot and obstacle:
                    return True
        return False

    def _check_success(self):
        # Called during initialization before all custom references exist.
        if not hasattr(self, "object_ids"):
            return False
        return self.metrics()["success"]

    def waypoint(self):
        p = self.object_positions()[self.target_object].copy()
        g = self.goal_positions[self.target_goal].copy()
        eef = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]].copy()
        self.max_lift = max(self.max_lift, p[2])
        # State-driven phases with timeouts for diagnostic failures.
        targets = [
            p + [0, 0, 0.14],
            p + [0, 0, 0.002],
            p + [0, 0, 0.002],
            np.array([eef[0], eef[1], 0.99]),
            g + [0, 0, 0.165],
            g + [0, 0, 0.013],
            g + [0, 0, 0.013],
            g + [0, 0, 0.17],
        ]
        target = targets[self.phase]
        return target, (-1.0 if self.phase in [0, 1, 6, 7] else 1.0)

    def advance_phase(self):
        target, _ = self.waypoint()
        eef = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        self.phase_ticks += 1
        reached = np.linalg.norm(target - eef) < (0.012 if self.phase in [1, 5] else 0.022)
        if self.phase in [2, 6]:
            reached = self.phase_ticks >= 12
        if reached and self.phase < 7:
            self.phase += 1
            self.phase_ticks = 0

    def reward(self, action=None):
        if not hasattr(self, "object_ids"):
            return 0.0
        target, grip = self.waypoint()
        eef = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        reach = 1 - np.tanh(12 * np.linalg.norm(eef - target))
        grasp = self._check_grasp(
            self.robots[0].gripper, [self.cube, self.other][self.target_object]
        )
        return float(reach + 0.25 * grasp + 0.2 * self.phase + 5 * self._check_success())

    def teacher_state(self):
        target, grip = self.waypoint()
        obs = self._get_observations()
        # All privileged fields explicitly confined to teacher interface.
        return np.r_[
            obs["robot0_eef_pos"],
            obs["robot0_gripper_qpos"],
            self.object_positions().ravel(),
            self.goal_positions[self.target_goal],
            target - obs["robot0_eef_pos"],
            np.eye(8)[self.phase],
            grip,
        ].astype(np.float32)

    def reference_action(self):
        target, grip = self.waypoint()
        eef = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        return np.r_[np.clip((target - eef) / 0.05, -0.8, 0.8), [0, 0, 0], grip].astype(np.float32)

    def step(self, action):
        obs, r, done, info = super().step(np.clip(action, -1, 1))
        self.advance_phase()
        self.collision_steps += int(self.forbidden_contact())
        positions = self.object_positions()
        self.wrong_target_any |= bool(positions[1 - self.target_object, 2] > 0.89)
        self.dropped_any |= bool((positions[:, 2] < 0.75).any())
        self.wrong_destination_any |= self.metrics()["wrong_destination"]
        info.update(self.metrics())
        return obs, r, done, info
