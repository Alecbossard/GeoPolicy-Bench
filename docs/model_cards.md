# Model cards — primary evaluations complete; VLA test pending

All weights/artifacts are local under ignored `artifacts/`. Principal models completed 8,000 updates each and 2,340 closed-loop test rollouts. Full resource/resume gates and the VLA final test are still running; see PROGRESS.md. Lower offline loss is not task success. Data, contracts and recipe are documented separately; this card records the actual adaptations.

## Privileged teacher

Stable-Baselines3 PPO MLP, 26D privileged state and 7D normalized OSC/gripper action, CPU optimization. Scripted 8-phase curriculum exposes current waypoint/phase/target object truth. 64 scripted bootstrap episodes initialize the actor with2500 supervised updates; selected checkpoint then receives2048 genuine PPO transitions. A continuation to34816 transitions is preserved as a failed fine-tuning experiment (latest validation2/20). Selected checkpoint validation17/20 on one20-scene set; independent20-scene controls: initial0/20, scripted20/20, BC19/20, selected PPO19/20. No statistically supported PPO gain is claimed. Task sequencing is scripted, not learned end to end.

## Compact ACT adaptation

1,051,239 parameters, randomly initialized CNN/CVAE/Transformer. Two RGB views at64px,23D robot state,explicit4D object/receptacle instruction semantics. Horizon8/execute2,common normalized actions. Final KL weight1;no pretrained visual/language backbone. Three seeds each completed8000 AdamW updates/batch32 using the same200 demonstrations as diffusion;EMA checkpoint selected on21 validation episodes. This is an independent compact ACT-inspired implementation,not exact ACT paper reproduction or unchanged LeRobot ACT. No metric depth input,so depth/extrinsic corruption does not change its inputs. Checkpoint includes optimizer/scheduler/normalization/RNG and batch generator state for continuation.

The CNN preserves 8×8 spatial features per camera, with learned positional embeddings for 128 image tokens. CVAE latent16 uses two encoder layers; the action decoder has two layers, width128 and4 attention heads. Training minimizes masked action L1 plus KL; inference uses latent zero and executes the first2 actions without temporal ensembling. Despite lower validation loss (seed0 minimum0.04418), final nominal success is0/20 for each of the3 seeds. Under missing fixed view, seed2 achieves1/20; all other ACT cells are0/20. This is a negative result for this compact recipe, not a conclusion about upstream ACT generally.

## Compact 3D diffusion adaptations

Approximately1.09M parameters, same architecture, demonstrations, points512, optimizer/batch budget, actions and conditioning for mono/fusion. XYZRGB point encoder with validity masks and learned attention/max pooling; per-frame calibrated wrist transform. Fusion combines fixed+wrist points in Panda base before the shared sampling budget. Color attention includes a declared fixed RGB contrast prior for the four known color words; it uses sensor RGB only, never simulator segmentation/object poses. This engineered prior limits generalization claims.

An early epsilon-prediction pilot (40 demos,1000 updates) is retained and failed20/20 nominal validation rollouts despite lower loss. A sample-prediction/legacy-prior pilot achieved2/20. Final recipe was frozen before test:chroma40 normalized RGB contrast with dark-intensity attenuation,clean-action prediction,100 cosine DDPM timesteps,10 DDIM iterations,horizon8/execute2. Three mono and three fusion seeds each completed8000 updates/batch32/200 same demonstrations,with identical state/action normalization and1,088,455 parameters. The first final-recipe fusion checkpoint achieved6/20 independent validation successes;final test is separate. Compact models use FP32. Short temporal FiLM residual denoiser and explicit proprio/instruction conditioning;not exact DP3,not a foundation model,no pretrained weights.

## Principal measured behavior

| Policy | Nominal success, seed0 /1 /2 | Fixed view missing, seed0 /1 /2 | Nominal episodes per seed |
| --- | --- | --- | --- |
| Mono diffusion | 24% /37% /30% | 0% /0% /3% | 100 |
| Fusion diffusion | 27% /34% /31% | 18% /19% /14% | 100 |
| Compact ACT | 0% /0% /0% | 0% /0% /5% | 20 |

Fusion−mono: nominal +0.33pp (descriptive95% crossed-bootstrap interval −5 to+5); missing fixed view +16pp (+9 to+23). Occlusion +5.67pp (−2 to+13.33) remains uncertain. Depth/extrinsic cells have only20 scenes per seed. Three seeds and a narrow four-instruction scene family limit generalization. Full raw metrics and error rates: `results/benchmark_summary.json`, `results/per_seed.csv` and `results/raw/student_rollouts.csv`.

## SmolVLA

Authentic `lerobot/smolvla_base`450,046,176 total parameters,LeRobot0.4.4 official model and image/text preprocessing. VLM frozen,action expert and allowed projections trainable (99,880,992 parameters),two RGB128 views resized/padded512px,actual text and robot state;no depth. Horizon reduced to8,final execute2;state/action dimensions padded to32 by upstream. Strict full base checkpoint loading,with redundant standalone VLM weight loading disabled. Real batch1 optimizer pilot passed with2.08GB peak allocated/2.19GB reserved.

Actual fine-tuning:500 optimizer updates,accumulation4,batch1,200same demonstrations,one training seed;637.08s optimization window,best offline loss.07831,probe parameter L2 change.609375. Native upstream trainable modules use96,607,440 BF16 and3,273,552 FP32 parameters;no independently added autocast/scaler. Peak allocated2.086GB/reserved2.233GB. First two validation rollouts0/2 success,~424ms policy median per chunk:does not sustain a20Hz real-time action stream with execute2. Final20-scene nominal test and conditional OOD budget are exploratory;not a budget-matched architecture comparison to8000-update compact models. Pinned model/tokenizer/config revisions and Apache-2.0 sources retained.
