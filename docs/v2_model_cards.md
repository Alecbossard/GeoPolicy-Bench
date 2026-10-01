# V2 model and comparison contract

These are independent compact implementations for a local simulated Panda task.
No exact paper reproduction, official benchmark score, pretrained vision or
physical-robot capability is claimed. Original V1 implementations remain intact.

| Family | Selected input | Prediction and execution | Training change |
| --- | --- | --- | --- |
| Diffusion | 512 calibrated XYZRGB points, fixed/wrist/fusion; four causal robot snapshots (92 values); four semantic tokens | Eight normalized 7D OSC actions, clean-action DDPM training with 100 steps, ten-step DDIM; execute two actions | Weighted six-channel motion MSE plus separate gripper-sign BCE |
| ACT RGB | Two 64×64 RGB images, four causal robot snapshots, four semantic tokens | Eight action chunks from a width-128, two-layer Transformer decoder; execute two actions | Optimize the exact zero-latent deployment path; unused CVAE modules frozen; motion MSE plus gripper-sign BCE |
| Rejected ACT 3D pilot | Fused points, four causal robot snapshots, four tokens | PointCondition and one projected memory token into a two-layer width-128 chunk Transformer | Deterministic 3D adaptation with motion MSE/gripper BCE; not official RGB ACT |

The RGB model's shared random CNN produces 128 spatial tokens (64 per image),
with learned positions. It has no pretrained image backbone. The original V1 ACT
used a future-action-conditioned CVAE posterior during training and zero latent
at inference. The posterior remains an offline oracle for diagnosis only. V2
freezes its unused modules and trains the deployed path directly. Total and
trainable parameter counts differ and are recorded per checkpoint.

An additional validation-only `act_instruction_broadcast` pilot adds the existing
learned instruction embedding to every visual token, without adding parameters.
It tests a hypothesis motivated by weak first-frame instruction sensitivity;
that sensitivity is not a demonstrated root cause. The initial six corrected
pilots were followed by this adaptive seventh pilot, all at 3,000 updates and
the same demonstrations/actions. It scored 0/20 stable placements and was not
selected. After it, ACT confirmation uses fresh seeds 100320–100339; diffusion
retains 100280–100299. The earlier ACT confirmation remains separately archived.

Diffusion uses point MLP features, learned instruction-conditioned attention,
max pooling, explicit XYZRGB attention moments and a FiLM temporal denoiser.
The six view/prior variants share architecture, parameters, optimization,
normalization and point budget. The manual prior adds `40 × normalized chromatic
contrast` to each of four attention heads for the known red/green/blue/yellow
colors. The ablation removes this additive bias only. Both variants still see RGB
and learn attention/moments; the ablation does not remove all color information.
Fixed-camera-loss corrupts the named fixed packet before selecting views, so
wrist-only observations remain present. Points are in robot-base coordinates;
state fields retain their original robot-observation coordinate convention,
identically in recorded and live processing.

Motion loss weights are `[1, 1, 1, 0.1, 0.1, 0.1]` on normalized translation and
rotation channels. A 0.1-weight auxiliary normalized-gripper MSE supervises the
seventh diffusion output, which remains part of the internal DDIM trajectory even
when a separate classifier supplies the physical command. Gripper labels use the sign of the denormalized demonstration
command. Inference decodes its logit to physical ±1 and converts back to action
normalization before the common evaluator denormalizes/clips the command. Panda's
backend uses the sign for incremental opening/closing. Future chunks near an
episode end are masked. History uses current state first, then past snapshots,
with the first state repeated for padding; snapshots are copied to avoid mutable
MuJoCo observation aliasing.

All final models use 200 identical successful demonstration prefixes extended by
30 recorded actions from the same preserved PPO teacher, plus 21 recorded
validation trajectories. Original state/action normalization is frozen across
all V2 variants. Every final run uses 8,000 updates, batch 32, AdamW 3e-4, weight
decay 1e-5, cosine minimum LR 3e-5 and EMA 0.995, on three seeds. Best checkpoints
are selected by recorded validation loss at fixed 500-update intervals. The
common architecture/recipe was chosen beforehand by separate closed-loop
validation. No test result selects a model or checkpoint.

ACT and diffusion differ in modality, capacity, inductive biases, trainable
modules and inference sampling. Their common data/actions/update budget is a
control, not a capacity-matched architecture comparison. The binary-gripper and
history pilots also change loss weighting; their combined improvement cannot be
attributed solely to binary grip or history. Dataset-continuation-only versus
original-data seed-0 pilots form the narrow matched data control. Three seeds
apply to final view/prior comparisons and selected ACT; targeted pilots are
single-seed and remain exploratory.

Truth object/goal poses and teacher phases are excluded from every student
input. A four-value vector concatenates two one-hot pairs: red/green target and
blue/yellow destination. The text sentence is rendered from those labels;
there is no learned text encoder. This is semantic task conditioning across
four known combinations, not open-vocabulary language understanding.
Evaluation uses truth only for scoring stages, errors, contacts and sustained
placement. Student actions continue during the stability dwell without teacher
waypoints, automatic gripper release or evaluator intervention.

Weights are local trusted artifacts. Manifests, checkpoint hashes, raw failures,
recorded/live input checks and seed-level results accompany the report. The
demonstration exports only frozen EMA weights, normalization and configuration;
optimizer and full resume state stay in main checkpoints. V1 PPO/SmolVLA results
and their limitations are unchanged; see the original report and notices.
