# Third-party sources and distribution scope

Project code: MIT, [LICENSE](../LICENSE). Compact ACT/diffusion code is an independent adaptation; references below are credited, not claimed as exact reproductions. Installed libraries and model weights retain their own licenses. This repository does not vendor their runtime source or pretrained weights.

| Source | License | Used for |
|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco/blob/main/LICENSE) | Apache-2.0 | Physics and official Windows bindings |
| [robosuite](https://github.com/ARISE-Initiative/robosuite/blob/master/LICENSE) | MIT and bundled asset notices | Panda, assets, OSC, rendering; custom task derives from Lift |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-style and bundled third-party notices | Optimization/inference |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) | MIT | Actual PPO implementation |
| [LeRobot](https://github.com/huggingface/lerobot/blob/main/LICENSE) | Apache-2.0 | Native dataset writer and authentic SmolVLA |
| [ACT](https://github.com/tonyzhaozh/act) | See upstream MIT license | CVAE/action-chunk architectural reference |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/blob/main/LICENSE) | MIT | Conditional action-diffusion reference |
| [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy) | MIT | Point-cloud diffusion architecture reference |
| [SmolVLA base](https://huggingface.co/lerobot/smolvla_base) | Apache-2.0 model card | Pinned authentic pretrained base |
| [SmolVLM2 500M](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | Apache-2.0 model card | Base VLM configuration/tokenizer, pinned revision |
| [ONNX](https://github.com/onnx/onnx) / [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Apache-2.0 / MIT | FP32 temporal-denoiser export and runtime |

The VLA environment's robosuite wheel only changes dependency metadata. Upstream runtime/assets/license files remain byte-identical; the build manifest records both wheel hashes. Do not use that wheel to claim support for excluded Mink whole-body IK. A future dataset/weights release must retain upstream asset/model license notices and the dataset/model cards. No external publication has been performed.

Dependency inventories are pinned in both requirements locks; the official version/license audit is `docs/upstream_audit.json`. Installation fetches normal upstream distributions with their bundled notices. No access token or secret is required for these public sources.
