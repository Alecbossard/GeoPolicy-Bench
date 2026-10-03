"""One physical cube, one tray, fixed instruction; same Panda OSC and cameras."""

import numpy as np
from geopolicy.environment import SelectPlace
from robosuite.environments.manipulation.lift import Lift


class SinglePlace(SelectPlace):
    def _load_model(self):
        super()._load_model()
        # Remove complete bodies before compiling physics: no hidden distractor.
        for body in list(self.model.worldbody):
            if body.get("name") in (self.other.root_body, "receptacle_1", "distractor"):
                self.model.worldbody.remove(body)

    def _setup_references(self):
        Lift._setup_references(self)
        self.object_ids = [self.cube_body_id]
        self.goal_body_ids = [self.sim.model.body_name2id("receptacle_0")]

    def _reset_internal(self):
        Lift._reset_internal(self)
        slot = int(self.rng_scene.integers(2))
        position = np.array([0.04, -0.08 if slot == 0 else 0.08, 0.83])
        position[:2] += self.rng_scene.uniform([-0.045, -0.025], [0.045, 0.025])
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.r_[position, [1, 0, 0, 0]])
        goal_slot = int(self.rng_scene.integers(2))
        goal = np.array([-0.10, -0.18 if goal_slot == 0 else 0.18, 0.825])
        goal[:2] += self.rng_scene.uniform([-0.025, -0.014], [0.025, 0.014])
        self.goal_positions[0] = goal
        self.sim.model.body_pos[self.goal_body_ids[0]] = [goal[0], goal[1], 0.8]
        self.sim.forward()
        self.phase = self.phase_ticks = 0
        self.max_lift = 0.0
        self.collision_steps = 0
        self.wrong_target_any = self.wrong_destination_any = self.dropped_any = False

    def reset_scene(self, seed, object_id=None, goal_id=None):
        assert object_id in (None, 0) and goal_id in (None, 0)
        return super().reset_scene(seed, 0, 0)

    def reward(self, action=None):
        return 0.0  # no RL is trained; teacher phase is diagnostic/collection only

    def metrics(self):
        position = self.object_positions()[0]
        goal = self.goal_positions[0]
        error = float(np.linalg.norm(position[:2] - goal[:2]))
        grasp = self._check_grasp(self.robots[0].gripper, self.cube)
        return {
            "success": bool(
                error < 0.043
                and 0.816 < position[2] < 0.855
                and not grasp
                and self.max_lift > 0.89
            ),
            "placement_error_m": error,
            "max_lift_m": self.max_lift,
            "collision": self.collision_steps > 0,
            "collision_steps": self.collision_steps,
            "dropped": bool(position[2] < 0.75),
        }

    def step(self, action):
        obs, reward, done, info = Lift.step(self, np.clip(action, -1, 1))
        self.advance_phase()  # never used to construct or override student commands
        self.collision_steps += int(self.forbidden_contact())
        info.update(self.metrics())
        return obs, reward, done, info
