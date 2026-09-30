import numpy as np
import pytest
from geopolicy.interfaces import RGBDFrame, CartesianCommand


def test_command_units_clipping_and_invalid_rejection():
    c = CartesianCommand.from_normalized([2, -2, 0, 1, 0, -1, -1])
    np.testing.assert_allclose(c.delta_xyz_m, [0.05, -0.05, 0])
    np.testing.assert_allclose(c.delta_rotation_axis_angle_rad, [0.5, 0, -0.5])
    assert c.gripper_velocity_normalized == -1
    with pytest.raises(ValueError):
        CartesianCommand.from_normalized([np.nan] * 7)


def test_calibration_rotation_must_not_be_a_reflection():
    f = RGBDFrame(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), np.eye(3), np.eye(4), 0)
    f.validate()
    f.base_from_camera[0, 0] = -1
    with pytest.raises(ValueError):
        f.validate()
