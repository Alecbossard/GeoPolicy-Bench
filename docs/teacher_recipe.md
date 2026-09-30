# Executed BC-initialized PPO recipe

The teacher is a privileged low-level actor within a scripted eight-phase curriculum. It is not an end-to-end learned task planner. Curriculum state and target poses are available only to the teacher/reward/evaluation, and never appear in student batches.

## State, curriculum and actions

The 26-dimensional teacher state concatenates world end-effector position (3), finger joint positions (2), both cube world positions (6), selected tray position (3), current waypoint error (3), phase one-hot (8), and desired gripper sign (1). It predicts the same clipped 7-dimensional OSC position/rotation/gripper command as students. Controller settings and frames are in [contracts](contracts.md).

| Phase | Waypoint | Gripper | Transition |
| --- | --- | --- | --- |
| 0 | Selected cube +14 cm z | Open | Within 2.2 cm |
| 1 | Selected cube +2 mm z | Open | Within 1.2 cm |
| 2 | Same grasp waypoint | Close | 12 control ticks elapsed |
| 3 | Current end-effector xy, world z=0.99 m | Close | Within 2.2 cm |
| 4 | Selected tray +16.5 cm z | Close | Within 2.2 cm |
| 5 | Selected tray +1.3 cm z | Close | Within 1.2 cm |
| 6 | Same release waypoint | Open | 12 control ticks elapsed |
| 7 | Selected tray +17 cm z | Open | Last phase |

The diagnostic script commands `clip(waypoint_error /0.05, −0.8,0.8)` translation, zero rotation and the phase gripper sign. Scripted bootstrap data is labeled separately. Phase advancement depends on position or the explicit grasp/release timer; students execute their predicted chunks directly with no script-driven action replacement. Curriculum bookkeeping still updates in the simulator but is not a student input.

## Reward and termination

Every transition uses:

`r = 1 − tanh(12 × ||eef − waypoint||) +0.25 × grasp +0.2 × phase +5 × success`

`grasp` is simulator contact with the selected cube. `success` is the instruction-correct lift/placement/release condition in the contracts. No drop/collision penalty is included in this recipe; those metrics are recorded separately. PPO terminates on success and truncates at 220 control steps (11 simulated seconds). Training scene seeds are sampled only from `[0,100000)`; validation and final test ranges are disjoint.

The dense phase reward is paid on every transition. Its interaction with early success termination can favor lingering in a late phase rather than completing quickly. This is a plausible reward-design limitation, not a tested explanation for the negative PPO continuation; no reward ablation was executed. The original reward and trained checkpoints are retained rather than changed using final test outcomes.

## Optimization and measured controls

Stable-Baselines3 2.7.0 PPO, CPU, seed 0, separate actor/critic MLPs `[128,128]`, initial log standard deviation −2.7, learning rate 1e−4, rollout length 1,024 transitions, minibatch 128, 5 epochs, gamma 0.98 and GAE lambda 0.95. The actor was first initialized with 2,500 supervised updates from 64 successful scripted bootstrap episodes. The selected checkpoint then received 2,048 actual on-policy PPO transitions; separated parameter audits measure actor ΔL2=0.060520 and critic/other ΔL2=1.018540 from BC.

Selection was on validation: the pilot achieved 17/20, while a continuation to 34,816 total transitions ended at 2/20 and its best validation was 16/20. The selected pilot produced all 274 raw collection episodes. On 100 independent final test scenes: initial 0/100, script 100/100, BC 82/100 and selected PPO 82/100. This verifies real RL updates and usable teacher data, with no demonstrated PPO success gain. The regression is retained as an executed negative result.

Checkpoints save SB3 optimizer state and separate Python/NumPy/Torch RNG files. Resume starts a new episode; bit-exact interrupted physics replay is not claimed. A separate non-selected 1,024-transition continuation profiles throughput/memory, and is not mixed into selected training steps or demonstration provenance. See `results/teacher_update_audit.json`, `results/teacher_resource_profile.json`, raw controls and [final report](final_report.md).
