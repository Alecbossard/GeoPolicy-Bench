# Architecture and information boundaries

```mermaid
flowchart LR
  Sim[MuJoCo Panda selection / placement] --> RGBD[Fixed + wrist RGB-D]
  Sim --> Robot[Robot proprioception 23D]
  Text[Instruction: object + destination] --> Student
  RGBD --> Geometry[Metric deprojection / moving extrinsics / crop / voxel /512 XYZRGB]
  Geometry --> Student[Compact mono or fusion diffusion]
  Robot --> Student
  RGBD --> RGB[Two RGB64 images]
  RGB --> ACT[Compact ACT / CVAE]
  Robot --> ACT
  Text --> ACT
  Student --> Chunk[8-step normalized action chunk]
  ACT --> Chunk
  Chunk --> Execute[Execute2 /7D OSC + gripper /20Hz simulator]
  Execute --> Sim
  Truth[Object truth + scripted curriculum] --> PPO[BC-initialized PPO teacher]
  PPO --> Data[Lossless HDF5 demonstrations]
  Data --> Train[Offline student optimization]
  Train --> Student
  Truth --> Metrics[Instruction-correct evaluation only]
```

No truth pose, segmentation or curriculum-phase edge enters students. Compact language conditioning is an explicit four-dimensional semantic encoding; SmolVLA uses actual text and official preprocessing. ACT/SmolVLA have RGB and robot state, no depth. The teacher is privileged and sequences tasks through a scripted curriculum, so it is a control/data source rather than a fair visual baseline.

All students share clipped7D Cartesian/gripper commands. Learned chunks are executed directly through the validated OSC controller; no object-aware action conversion is applied. Inference computes moving wrist calibration from the robot/simulator camera pose, corresponding to a future calibrated FK transform, not target object truth. See [contracts](contracts.md) and [future physical transfer](calibration_and_transfer.md).

Training is serial with full optimizer/scheduler/normalization/RNG checkpoints. Test evaluation has independent scene seeds and persists each episode. ONNX exports only the temporal denoiser; point encoding, DDIM iteration and sensor preprocessing remain PyTorch/NumPy. Physical hardware transfer, Isaac and Jetson are outside executed results.
