# Dataset card — GeoPolicy SelectPlace v1

Locally generated MuJoCo/robosuite Panda demonstrations, simulation only. Two RGB-D views and proprioception, instructions selecting one of two colored cubes and one of two colored receptacles. The physical scene geometry is randomized by seed, including color-to-slot permutations. Gray distractor; no claim of broad object/language diversity.

Collection: 250 training-scene episodes, 208 successful and 42 failures; 24 disjoint validation-scene episodes, 21 successful and 3 failures. Main models use the same first 200 successful training episodes and all 21 successful validation episodes. Selection and per-file audit hashes are frozen in `configs/dataset_manifest.json`. Approximately 2.09 GB of raw compressed HDF5 across all episodes. No test scene is included.

Provenance: actual rollouts of the selected PPO checkpoint, which was initialized by supervised BC from a scripted controller. Checkpoint SHA256 `384576a980ee55a6969d1fe00caf85cf0323dccbcf5f1fcc8562ea4d9e607d63`. Do not describe the BC bootstrap trajectories as PPO data, or describe this teacher as fully learned task sequencing. The phase curriculum is scripted privileged teacher information. Scripted diagnostic data in `artifacts/data_smoke` and `artifacts/reference_pilot` is not part of the main student dataset.

Each HDF5 episode contains lossless RGB uint8, metric depth float32, intrinsics, per-frame base-from-camera extrinsics, world-from-base transform, XYZRGB points/validity masks, robot state, normalized 7D Cartesian/gripper actions, semantic instruction tokens and timestamps. A final RGB-D endpoint is separate from pre-action frames. Metadata includes full instruction, scene seed/config, success/failure, provenance and teacher hash. Truth target IDs in metadata are only for audit/instruction semantics, never object poses or teacher phases in student inputs.

Splits are entire scene episodes by seed ranges: train [0,100000), validation [100000,200000), test >=200000. Train collection uses seeds1000–1249, validation100000–100023. Test is reserved for final behavior experiments. Adjacent frames cannot cross splits. Teacher learning never uses test scenes. Successful-only training introduces survivorship bias; failures are retained for diagnosis. Seeds represent a narrow randomized scene family, not held-out shape categories.

RGB-D compression is lossless; no lossy depth video. The LeRobot export will represent compatible RGB/state/action/instruction features, with indexed lossless depth/point sidecars rather than pretending metric depth is a video image.

Artifacts remain local and ignored by Git. Suggested publication: an archive/Hub dataset plus this card and manifests after explicit user authorization; none has been published. Synthetic data contains no real-person recordings. Upstream asset/software licensing must accompany distribution. See technical decisions and contracts.
