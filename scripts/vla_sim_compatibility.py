"""Verify unchanged Panda scene under dependency-only NumPy2 OSC profile."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
import numpy as np
import mujoco
import robosuite
from geopolicy.environment import SelectPlace
from geopolicy.sensors import camera_packet

env = SelectPlace(cameras=True)
obs = env.reset_scene(121)
packet = camera_packet(env, obs)
model = env.sim.model.get_xml()
package_path = Path(robosuite.__file__).parent
model = model.replace("\\", "/").replace(package_path.as_posix(), "ROBOSUITE")
report = {
    "numpy": np.__version__,
    "mujoco": mujoco.__version__,
    "robosuite_runtime": robosuite.__version__,
    "model_xml_sha256": hashlib.sha256(model.encode()).hexdigest(),
    "objects": env.object_positions().tolist(),
    "goals": env.goal_positions.tolist(),
    "fixed_points": packet["agentview"]["points"].tolist(),
    "wrist_points": packet["robot0_eye_in_hand"]["points"].tolist(),
    "observation_keys": list(obs),
    "eef": obs["robot0_eef_pos"].tolist(),
}
for step in range(15):
    obs, _, _, _ = env.step(env.reference_action())
report["eef_after15"] = obs["robot0_eef_pos"].tolist()
Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/vla_sim_compatibility.json").with_suffix(
    ".xml"
).write_text(model)
Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/vla_sim_compatibility.json").write_text(
    json.dumps(report, indent=2)
)
print({k: v for k, v in report.items() if "points" not in k})
env.close()
