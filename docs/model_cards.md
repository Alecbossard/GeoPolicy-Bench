# Model cards — GeoPolicy SelectPlace v1

All weights and full artifacts are local under ignored `artifacts/`. Nine principal models completed 8,000 updates each and 2,340 closed-loop test rollouts. SmolVLA testing and final resource/reproduction gates are complete; see the [final report](final_report.md). Lower offline loss does not establish task success. This card records the actual model adaptations and measured behavior.

## Privileged teacher

Stable-Baselines3 PPO MLP, 26-dimensional privileged state and seven normalized OSC/gripper actions, optimized on CPU. A scripted eight-phase curriculum exposes waypoint, phase and target object truth. The actor receives 2,500 supervised updates from 64 scripted bootstrap episodes, then 2,048 genuine PPO transitions. A continuation to 34,816 total transitions is preserved as a failed fine-tuning experiment: its latest validation success fell to 2/20. The selected pilot scored 17/20 on its selection set; independent validation controls scored initial 0/20, scripted 20/20, BC 19/20 and PPO 19/20. Final paired test controls scored initial 0/100, scripted 100/100, BC 82/100 and PPO 82/100. PPO parameter learning is verified; a success gain is not demonstrated. Task sequencing remains scripted. See [teacher recipe](teacher_recipe.md).

## Compact ACT adaptation

1,051,239 parameters in a randomly initialized CNN/CVAE/Transformer. Inputs: two 64-pixel RGB views, 23-dimensional robot state and four explicit object/receptacle instruction semantics. The eight-action horizon executes two actions per prediction, using the common normalized action contract. KL weight is 1; there is no pretrained backbone. Three seeds each completed 8,000 AdamW updates at batch 32 on the same 200 demonstrations as diffusion; EMA checkpoints were selected on 21 validation episodes. This is an independent compact ACT adaptation. It is not an exact paper reproduction or unchanged LeRobot ACT. Depth and extrinsic corruption do not change its RGB inputs. Checkpoints preserve optimizer, scheduler, normalization and all relevant RNG state.

The CNN preserves 8×8 spatial features per camera, with learned positional embeddings for 128 image tokens. The 16-dimensional CVAE latent uses two encoder layers. The action decoder has two layers, width 128 and four attention heads. Training minimizes masked action L1 plus KL; inference uses latent zero without temporal ensembling. Despite decreasing validation loss (seed 0 minimum 0.04418), final nominal success is 0/20 for every seed. With the fixed view missing, seed 2 achieves 1/20; all other ACT cells score 0/20. This negative result applies to this compact recipe.

## Compact 3D diffusion adaptations

Each variant has 1,088,455 parameters and shares architecture, demonstrations, 512-point budget, optimizer, actions and instruction conditioning. The XYZRGB encoder uses validity masks, learned attention and max pooling. Fusion combines fixed and wrist points in the Panda base frame with a calibrated wrist transform for each frame. Color attention includes a declared fixed RGB contrast prior for the four known color words. It uses sensor RGB; simulator segmentation and object poses are excluded. This engineered prior limits generalization claims.

An early epsilon-prediction pilot (40 demos, 1,000 updates) failed all 20 nominal validation rollouts despite decreasing loss. A sample-prediction/legacy-prior pilot achieved 2/20. The final recipe was frozen before test: `chroma40` normalized RGB contrast with dark-intensity attenuation, clean-action prediction, 100 cosine DDPM timesteps, 10 DDIM iterations, horizon eight and execution of two actions. Three mono and three fusion seeds each completed 8,000 updates at batch 32 on the same 200 demonstrations, with identical normalization. The first final-recipe fusion checkpoint achieved 6/20 independent validation successes; final test scenes are separate. Models use FP32, a temporal FiLM residual denoiser and explicit robot state/instruction conditioning. They are independent DP3-inspired adaptations without pretrained weights.

## Principal measured behavior

| Policy | Nominal success, seed0 /1 /2 | Fixed view missing, seed0 /1 /2 | Nominal episodes per seed |
| --- | --- | --- | --- |
| Mono diffusion | 24% /37% /30% | 0% /0% /3% | 100 |
| Fusion diffusion | 27% /34% /31% | 18% /19% /14% | 100 |
| Compact ACT | 0% /0% /0% | 0% /0% /5% | 20 |

Fusion minus mono: nominal +0.33 percentage points (descriptive 95% crossed-bootstrap interval −5 to +5); missing fixed view +16 points (+9 to +23). Occlusion +5.67 points (−2 to +13.33) remains uncertain. Depth/extrinsic cells have 20 scenes per seed. Three seeds and a narrow scene family limit generalization. Full metrics and error rates are in `results/benchmark_summary.json`, `results/per_seed.csv` and `results/raw/student_rollouts.csv`.

Counterfactual tests keep the initial sensor observations identical while changing all four instructions. None of the six mono/fusion model runs completes all four instructions on any of the ten scenes. First actions change with the semantic tokens, but reliable completion under all instruction changes is not demonstrated.

## SmolVLA

Authentic `lerobot/smolvla_base`: 450,046,176 total parameters, official LeRobot 0.4.4 model and image/text preprocessing. The VLM is frozen; the action expert and allowed projections are trainable (99,880,992 parameters). Inputs are two 128-pixel RGB views resized/padded to 512 pixels, actual text and robot state, without depth. The action horizon is eight, with two actions executed; upstream pads state/actions to 32 dimensions. The full pinned base checkpoint loads strictly, with redundant standalone VLM loading disabled. A real batch-one optimizer pilot passed with about 2.08 GB allocated and 2.19 GB reserved.

Actual fine-tuning: 500 optimizer updates, accumulation four, batch one, the same 200 demonstrations and one training seed. The optimization window took 637.08 seconds; best offline loss was 0.07831 and probe parameter L2 change was 0.609375. Native upstream trainable modules contain 96,607,440 BF16 and 3,273,552 FP32 parameters; no separate autocast/scaler was added. Peak allocation was 2.086 GB, with 2.233 GB reserved. The two validation rollouts scored 0/2 success, with about 424 ms median policy latency per chunk. With two actions executed at 20 Hz, this does not sustain real-time control. The final 20-scene nominal test and conditional OOD budget are exploratory, with a different optimization budget from the compact models. Pinned base/tokenizer/config revisions and Apache-2.0 credits are retained.

The completed nominal test scores 0/20, with five collision flags and mean terminal XY error 21.16 cm. Mean episode policy P50/P95 is 612.53/744.57 ms; these means include the original and resumed evaluation sessions. OOD is deferred under the predeclared zero-nominal-success gate, rather than a memory failure. A final optimizer step with loaded AdamW moments and both RGB-D renderers alive passed with 4,166 MiB of whole-device GPU memory free. This exploratory failure applies to the executed adaptation and budget.
