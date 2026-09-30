# Model cards — provisional, benchmark incomplete

All current weights/artifacts are local under ignored `artifacts/`. Measured outcomes belong in the final report after closed-loop tests; do not substitute lower loss for task success.

## Privileged teacher

Stable-Baselines3 PPO MLP, 26D privileged state and 7D normalized OSC/gripper action, CPU optimization. Scripted 8-phase curriculum exposes current waypoint/phase/target object truth. 64 scripted bootstrap episodes initialize the actor with2500 supervised updates; selected checkpoint then receives2048 genuine PPO transitions. A continuation to34816 transitions is preserved as a failed fine-tuning experiment (latest validation2/20). Selected checkpoint validation17/20 on one20-scene set; independent20-scene controls: initial0/20, scripted20/20, BC19/20, selected PPO19/20. No statistically supported PPO gain is claimed. Task sequencing is scripted, not learned end to end.

## Compact ACT adaptation

1,051,239 parameters, randomly initialized CNN/CVAE/Transformer. Two RGB views at64px,23D robot state,explicit4D object/receptacle instruction semantics. Horizon8/execute2,common normalized actions. Final KL weight1;no pretrained visual/language backbone. Three seeds each completed8000 AdamW updates/batch32 using the same200 demonstrations as diffusion;EMA checkpoint selected on21 validation episodes. This is an independent compact ACT-inspired implementation,not exact ACT paper reproduction or unchanged LeRobot ACT. No metric depth input,so depth/extrinsic corruption does not change its inputs. Checkpoint includes optimizer/scheduler/normalization/RNG and batch generator state for continuation.

## Compact 3D diffusion adaptations

Approximately1.09M parameters, same architecture, demonstrations, points512, optimizer/batch budget, actions and conditioning for mono/fusion. XYZRGB point encoder with validity masks and learned attention/max pooling; per-frame calibrated wrist transform. Fusion combines fixed+wrist points in Panda base before the shared sampling budget. Color attention includes a declared fixed RGB contrast prior for the four known color words; it uses sensor RGB only, never simulator segmentation/object poses. This engineered prior limits generalization claims.

An early epsilon-prediction pilot (40 demos,1000 updates) is retained and failed20/20 nominal validation rollouts despite lower loss. A sample-prediction/legacy-prior pilot achieved2/20. Final recipe was frozen before test:chroma40 normalized RGB contrast with dark-intensity attenuation,clean-action prediction,100 cosine DDPM timesteps,10 DDIM iterations,horizon8/execute2. Three mono and three fusion seeds each completed8000 updates/batch32/200 same demonstrations,with identical state/action normalization and1,088,455 parameters. The first final-recipe fusion checkpoint achieved6/20 independent validation successes;final test is separate. Compact models use FP32. Short temporal FiLM residual denoiser and explicit proprio/instruction conditioning;not exact DP3,not a foundation model,no pretrained weights.

## SmolVLA

Authentic `lerobot/smolvla_base`450,046,176 total parameters,LeRobot0.4.4 official model and image/text preprocessing. VLM frozen,action expert and allowed projections trainable (99,880,992 parameters),two RGB128 views resized/padded512px,actual text and robot state;no depth. Horizon reduced to8,final execute2;state/action dimensions padded to32 by upstream. Strict full base checkpoint loading,with redundant standalone VLM weight loading disabled. Real batch1 optimizer pilot passed with2.08GB peak allocated/2.19GB reserved.

Actual fine-tuning:500 optimizer updates,accumulation4,batch1,200same demonstrations,one training seed;637.08s optimization window,best offline loss.07831,probe parameter L2 change.609375. Native upstream trainable modules use96,607,440 BF16 and3,273,552 FP32 parameters;no independently added autocast/scaler. Peak allocated2.086GB/reserved2.233GB. First two validation rollouts0/2 success,~424ms policy median per chunk:does not sustain a20Hz real-time action stream with execute2. Final20-scene nominal test and conditional OOD budget are exploratory;not a budget-matched architecture comparison to8000-update compact models. Pinned model/tokenizer/config revisions and Apache-2.0 sources retained.
