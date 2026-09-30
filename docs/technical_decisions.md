# Technical decisions (2026-09-30)

Authoritative scope: local 8 GB hardware revision, MuJoCo/robosuite, one agent.

## Sources, versions and licenses
The machine-readable official PyPI/GitHub audit is upstream_audit.json. Latest releases at lookup: MuJoCo 3.14.0, robosuite 1.5.2, PyTorch 2.14.0, LeRobot 0.6.1 (requires Python >=3.12), SB3 2.9.0, diffusers 0.40.0. Latest does not imply an established compatible stack.

| Component | Local selection | Source / license / adaptation |
|---|---|---|
| MuJoCo | 3.3.7, Python 3.11 | https://mujoco.readthedocs.io/en/stable/python.html and https://github.com/google-deepmind/mujoco; Apache-2.0; official Windows wheel and WGL rendering |
| robosuite | 1.5.2 | https://robosuite.ai/docs/installation.html and tag v1.5.2; MIT plus upstream notices (LICENSE inspected); Panda Lift and custom selection/placement environment, OSC Cartesian command |
| PyTorch | 2.7.1 CUDA 12.8 wheel | https://pytorch.org/get-started/previous-versions/; BSD-style distribution notices; driver 591.66 sufficient; no toolkit/driver installation |
| PPO | stable-baselines3 2.7.0 | https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html; MIT; privileged teacher observations and curriculum, no vision in teacher training |
| ACT | compact adaptation initially; LeRobot source inspected before implementation | https://huggingface.co/docs/lerobot/act and https://github.com/huggingface/lerobot; Apache-2.0; action chunks, CVAE/Transformer, explicit object/receptacle instruction tokens; no pretrained backbone initially |
| Diffusion Policy | compact independent adaptation | https://github.com/real-stanford/diffusion_policy head 5ba07ac6661db573af695b419a7947ecb704690f; MIT; conditional action denoising, short horizon |
| DP3 | architecture inspiration, not exact reproduction | https://github.com/YanjieZe/3D-Diffusion-Policy head 47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97; MIT; published ~10 GB configuration is unsuitable; compact XYZRGB+mask encoder, two-view geometric fusion, 512 points |
| SmolVLA | isolated Python 3.11, LeRobot 0.4.4 | https://huggingface.co/docs/lerobot/smolvla and https://huggingface.co/lerobot/smolvla_base; Apache-2.0 model/package; authentic optimization pilot passed, official image/text processing, frozen VLM/action expert trained |

## Compatibility and resource gates
robosuite primarily supports Linux/macOS; official docs also describe Windows WGL/DLL adaptations. Try native isolated Windows first; existing Ubuntu WSL is a fallback only. Keep fixes inside project/environment. WSL virtual free disk is not physical free disk. No WSL install, driver/system change, reboot, cloud spending or publication.
Native environment is intentionally separate from future LeRobot: newest LeRobot requires Python 3.12 and may impose torch/video constraints. Freeze a fully resolved installation after smoke passes. Store compressed metric depth and RGB losslessly at reduced resolution, episode/scene-disjoint splits. Do not claim official LIBERO scores, Isaac experience, physical transfer or VLA fine-tuning before execution.

## Scientific constraints
Main comparison changes only view selection/fusion; same points budget, action representation, architecture, instruction, data, optimization and seeds. RGB colors remain available. Students only observe RGB-D, calibrated transforms, proprioception and instruction; no object truth/teacher phase. Privileged teacher is an upper bound, not a fair visual baseline. Declare scripted/BC bootstrap if used. Select checkpoints on validation before unlocking test. Negative results are valid; loss reduction alone is not task success.

## Initial proposed settings (not yet measured)
20 Hz Cartesian OSC position/rotation deltas + gripper; 128 px RGB-D, 512 points, chunk horizon 8, small batches. One heavy GPU job at a time. Measure actual memory, throughput and disk before expanding. Main task takes priority over stacking and secondary variants.

### Installed resolution update
robosuite 1.5.2 pins mink 0.0.5, requiring NumPy <2. Initial proposed NumPy 2.2.6 was rejected by resolver; corrected to 1.26.4. Native WGL works without editing upstream DLL/rendering code. Installed full pins: ../requirements-lock.txt. CUDA optimization and dual RGB-D Lift gate passed; results are a runtime sanity check, not a trained policy.

### LeRobot isolation and SmolVLA loader
Selected LeRobot 0.4.4 (official PyPI metadata >=Python3.10) rather than latest 0.6.1 >=3.12, allowing cached Torch 2.7.1 Windows wheel reuse. Its rerun-sdk requires NumPy>=2, confirming isolated environment is necessary. VLA env uses NumPy 2.2.6. Official preprocessing API inspected: base config factory handles `type`, action chunks need an explicit batch dimension. A Windows commit-limit error1455 occurred during repeated full backbone loading alongside collection; no system change. Retry serially using official load_vlm_weights=False + strict full SmolVLA weights. All final model params must be loaded from the authentic checkpoint; no random-backbone substitution.

### Executed VLA solution
The serial retry passed real loss/backward/AdamW at batch 1, with 2.083 GB peak allocated and ~5.21 GB GPU free. A 500-update/accumulation-4 fine-tuning run then completed in 637 s (optimization window). Frozen authentic VLM, trainable action expert/projections: 99,880,992 of 450,046,176 parameters. Pinned checkpoint/tokenizer/backbone revisions are in `configs/pretrained_revisions.json`; strict full base load ensures no random visual backbone. Official image preprocessing includes 512 px resize/pad. First two validation rollouts failed; lower loss alone is insufficient.

SmolVLA and the same Panda OSC simulation must run in one process on this 16 GB host. `scripts/build_osc_sim_wheel.py` verifies the official wheel SHA and builds a metadata-only `robosuite1.5.2+geopolicyosc` wheel, removing optional `mink`/`qpsolvers` dependencies unused by Panda OSC. All 1,176 runtime/asset/license files remain byte-identical. NumPy 2 in the VLA environment is validated by an identical seed/control rollout: object/tray/eef positions and both point clouds match exactly; normalized XML hash identical after removing installation-path differences. Manifests/logs: `artifacts/compatibility_wheels/`, `artifacts/{main,vla}_sim_compatibility.json`, `artifacts/compatibility_compare.log`. This metadata change does not support unused whole-body IK controllers; it is not a modified physics simulator. No system/pagefile change.

### Frozen compact recipe and implementation-only updates
`configs/benchmark_protocol.json` freezes 200 successful demonstrations, 3 seeds per model,8,000 updates/batch 32,EMA selection on 21 validation episodes, 512 XYZRGB points,horizon8/execute2. Diffusion predicts clean actions using 100 cosine DDPM timesteps/10 DDIM steps. Fixed RGB chromatic contrast 40x plus learned attention handles the four known color words; dark-intensity attenuation and normalized RGB contrast use actual sensor colors only. This is a declared engineered inductive prior, not simulator truth or arbitrary-language grounding. ACT KL weight 1 and no pretrained backbone.

All recipe adjustments used validation scenes only. After freeze, HDF5 arrays were read once per episode to reduce overhead (same point representatives, exact equality test), default-zero view-dropout support was added for the predefined secondary ablation, and evaluators gained resume/CSV/ONNX support. Unaffected wrist observations are preserved exactly under fixed-view corruptions. These do not change the primary training recipe; final execution source hashes are recorded separately. Never tune against final test behavior.

CPU with two threads batch 1 inference was faster than CUDA dispatch for the compact pilot (~12.16 ms versus 30.40 ms median policy-only). Final compact closed-loop inference uses CPU consistently; RTX training and separate RTX batch 1 profiles remain part of the deliverable. GPU dispatch overhead is a hypothesis, not a verified profiler finding. Report simulation 20 Hz action timing separately from measured wall-clock throughput; SmolVLA ~424 ms per two executed actions does not sustain real-time 20 Hz on this machine.

### Resource scheduling
One heavy GPU job runs at a time. Multiple auxiliary Python processes briefly reduced Windows commit availability to~0.55GB despite~5.5GB physical RAM free; the auxiliary teacher-control process was stopped and tasks serialized. Training remained checkpointed. Guards stop a job for persistent>=90 C, disk <12 GiB, and (training) repeated Windows commit availability <512 MiB. Ordinary observed GPU temperatures remained~40–63C. This is resource scheduling, not evidence of hardware instability.
