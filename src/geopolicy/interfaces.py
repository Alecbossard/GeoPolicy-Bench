"""Future sensor/robot boundary; no physical robot connection is implemented."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    base_from_camera: np.ndarray
    timestamp_s: float

    def validate(self):
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("RGB must be HxWx3")
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("Depth/RGB alignment required")
        if self.intrinsic.shape != (3, 3) or min(self.intrinsic[0, 0], self.intrinsic[1, 1]) <= 0:
            raise ValueError("Positive calibrated intrinsics required")
        if self.base_from_camera.shape != (4, 4):
            raise ValueError("Extrinsic must be 4x4")
        r = self.base_from_camera[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(r), 1, atol=1e-4
        ):
            raise ValueError("Extrinsic rotation must be proper orthonormal")
        if not np.allclose(self.base_from_camera[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous transform")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("Invalid timestamp")


@dataclass
class CartesianCommand:
    delta_xyz_m: np.ndarray
    delta_rotation_axis_angle_rad: np.ndarray
    gripper_velocity_normalized: float

    @classmethod
    def from_normalized(cls, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Finite 7D command required")
        action = np.clip(action, -1, 1)
        return cls(action[:3] * 0.05, action[3:6] * 0.5, float(action[6]))


class SensorInterface(Protocol):
    def read_synchronized(self) -> tuple[dict[str, RGBDFrame], np.ndarray]: ...


class RobotInterface(Protocol):
    def command_cartesian(self, command: CartesianCommand) -> None: ...
    def stop(self) -> None: ...
