from geopolicy.v2.config import read_recipe
from geopolicy.v2.metrics import StableWindow, placement_valid


def test_instantaneous_placement_is_not_stable_success():
    window = StableWindow(1.0)
    assert not window.update(0.0, True)
    assert not window.update(0.95, True)
    assert not window.update(1.0, False)
    assert not window.update(1.1, True)
    assert not window.update(2.05, True)
    assert window.update(2.1, True)


def test_one_finger_contact_or_motion_blocks_released_stability():
    sample = dict(
        inside=True,
        height_valid=True,
        previously_lifted=True,
        finger_width=0.06,
        finger_contact=False,
        linear_speed=0.005,
        angular_speed=0.02,
        config=read_recipe()["stability"],
    )
    assert placement_valid(**sample)
    for key, value in [
        ("finger_contact", True),
        ("finger_width", 0.04),
        ("linear_speed", 0.03),
        ("angular_speed", 0.3),
        ("inside", False),
        ("previously_lifted", False),
    ]:
        assert not placement_valid(**dict(sample, **{key: value}))


def test_rotated_cube_rejects_center_only_false_positive():
    import numpy as np
    from geopolicy.v2.config import read_recipe
    from geopolicy.v2.metrics import cube_inside_tray

    cfg = read_recipe()["stability"]
    center = np.array([0.04, 0.0, 0.83])
    goal = np.array([0.0, 0.0, 0.813])
    assert np.linalg.norm(center[:2] - goal[:2]) < 0.043
    assert cube_inside_tray(center, np.eye(3), goal, cfg)
    angle = np.pi / 4
    rotated = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    assert not cube_inside_tray(center, rotated, goal, cfg)
