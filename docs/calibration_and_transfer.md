# Future physical transfer boundary

No physical robot, ROS2 transport, Jetson, Isaac Lab or sim-to-real execution has occurred in this project. The protocol interfaces in `interfaces.py` define synchronized metric RGB-D and normalized Cartesian commands; they do not connect to hardware.

For a future Panda experiment, first validate the actual robot controller's delta frame, rotation convention, gripper sign, update rate and workspace limits in a separate safe commissioning procedure. The simulator's 5 cm command scale must not be adopted as a physical command limit without hardware validation. Add actual joint/velocity/force limits, watchdog, emergency stop and timestamp freshness at that boundary.

Calibrate each camera's intrinsics and depth scale against a measured target. Verify RGB-depth registration, exposure timestamps and driver units (meters, not raw depth counts). Save image dimensions, distortion model, intrinsic matrix, scale, calibration residuals and capture settings. Do not pass distorted pixels to pinhole deprojection without rectification.

For the fixed camera, estimate base-from-camera using surveyed fiducial correspondences and independently verify at multiple positions/heights in the workspace. For the wrist, collect diverse robot end-effector poses and checkerboard observations for hand-eye calibration; solve eef-from-camera, then compute base-from-camera(t)=base-from-eef(t) @ eef-from-camera. Record tool/flange frame naming, xyzw/wxyz conversions and transforms at the synchronized capture time. Use independent withheld target locations to test errors and detect frame/sign mistakes.

Measure synchronization between fixed camera, wrist camera and robot state. Start with a stationary target, then move through the workspace and inspect disagreement between deprojected view clouds. Set a measured acceptable timestamp skew, test dropped frames, reject stale observations, and keep calibration uncertainty in deployment diagnostics. Simulation's known transforms and exact synchronization are not evidence that this is solved physically.

Replay recorded physical sensor packets offline through the preprocessing before commanding the robot. Verify RGB colors, metric geometry, masks, workspace clipping, state normalization and action contract; then test bounded closed-loop motions under a separate hardware protocol. Domain randomization and sensor corruption experiments only prepare hypotheses for this transfer.
