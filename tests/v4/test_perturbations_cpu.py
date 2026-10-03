"""Synthetic CPU invariants for the frozen V4 point-representation corruptions.

These tests do not render cameras, run a policy, or establish manipulation
success. Raw RGB/depth remain provenance; corruptions affect sampled XYZRGB.
"""

import copy
import json

import numpy as np
import pytest

from geopolicy.v4.perturbations import (
    CAMERAS,
    augmentation_condition,
    evaluation_rng,
    packet_digest,
    perturb,
)


def synthetic_packet(*, padded=True):
    """Two calibrated views with nonconstant rays, depth and point colors."""
    index = np.arange(512, dtype=np.float32)
    z = 1 + (index % 17) / 17 * 0.6
    camera_xyz = np.stack(
        ((index % 32 / 31 - 0.5) * 0.5 * z, (index // 32 / 15 - 0.5) * 0.36 * z, z),
        axis=1,
    )
    colors = np.stack(((index % 11) / 10, (index % 13) / 12, (index % 7) / 6), axis=1)
    packet = {}
    for view, camera in enumerate(CAMERAS):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = (0.2, -0.3, 0.4)
        if view:
            transform[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        points = np.concatenate(
            (camera_xyz @ transform[:3, :3].T + transform[:3, 3], colors), axis=1
        ).astype(np.float32)
        mask = np.ones(512, dtype=bool)
        if padded:
            mask[-16:] = False
            points[-16:] = 0
        packet[camera] = {
            "points": points,
            "mask": mask,
            "intrinsic": np.array(
                ((128, 0, 63.5), (0, 128, 63.5), (0, 0, 1)), np.float32
            ),
            "base_from_camera": transform,
            "rgb": np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3),
            "depth_m": np.ones((128, 128), dtype=np.float32),
        }
    return packet


def assert_packet_unchanged(packet, saved):
    for camera in CAMERAS:
        for field in saved[camera]:
            np.testing.assert_array_equal(packet[camera][field], saved[camera][field])


def test_absent_fixed_camera_keeps_wrist_and_source_unchanged():
    packet = synthetic_packet()
    saved = copy.deepcopy(packet)
    result = perturb(
        packet, {"family": "absent", "intensity": 1}, evaluation_rng(19, 0)
    )
    assert not result[CAMERAS[0]]["mask"].any()
    assert not result[CAMERAS[0]]["points"].any()
    for field in ("points", "mask", "intrinsic", "base_from_camera"):
        np.testing.assert_array_equal(
            result[CAMERAS[1]][field], saved[CAMERAS[1]][field]
        )
    assert_packet_unchanged(packet, saved)


@pytest.mark.parametrize(
    "condition",
    (
        {"family": "nominal", "intensity": 0},
        {"family": "absent", "intensity": 1},
        {"family": "occlusion", "intensity": 0.6},
        {"family": "depth", "intensity": 0.025},
        {"family": "missing", "intensity": 0.7},
    ),
    ids=("nominal", "absent", "occlusion", "depth", "missing"),
)
def test_invalid_masked_points_are_zero_and_never_reenabled(condition):
    packet = synthetic_packet()
    # Stale finite values behind false masks must not enter the policy.
    for camera in CAMERAS:
        packet[camera]["points"][-16:] = 37
    saved = copy.deepcopy(packet)
    result = perturb(packet, condition, evaluation_rng(20, 3))
    for camera in CAMERAS:
        points, mask = result[camera]["points"], result[camera]["mask"]
        assert points.shape == (512, 6) and points.dtype == np.float32
        assert mask.shape == (512,) and mask.dtype == np.bool_
        assert np.isfinite(points).all()
        assert not np.any(mask & ~saved[camera]["mask"])
        assert np.count_nonzero(points[~mask]) == 0
        np.testing.assert_array_equal(
            points[mask, 3:], saved[camera]["points"][mask, 3:]
        )
    assert_packet_unchanged(packet, saved)


@pytest.mark.parametrize("intensity", (0, 1))
def test_missing_points_endpoints_apply_to_both_cameras(intensity):
    packet = synthetic_packet()
    result = perturb(
        packet, {"family": "missing", "intensity": intensity}, evaluation_rng(21, 0)
    )
    for camera in CAMERAS:
        if intensity == 1:
            assert not result[camera]["mask"].any()
            assert not result[camera]["points"].any()
        else:
            np.testing.assert_array_equal(
                result[camera]["mask"], packet[camera]["mask"]
            )
            np.testing.assert_array_equal(
                result[camera]["points"], packet[camera]["points"]
            )


@pytest.mark.parametrize("invalid", ("nan", "point_shape", "mask_shape"))
def test_invalid_packet_is_rejected_before_policy_input(invalid):
    packet = synthetic_packet()
    fixed = packet[CAMERAS[0]]
    if invalid == "nan":
        # Non-finite values are rejected even in padding, before any corruption.
        fixed["points"][-1, 0] = np.nan
    elif invalid == "point_shape":
        fixed["points"] = fixed["points"][:-1]
    else:
        fixed["mask"] = fixed["mask"][:-1]
    with pytest.raises(AssertionError):
        perturb(packet, {"family": "nominal", "intensity": 0}, evaluation_rng(21, 0))


def test_paired_missing_intensities_have_nested_masks_on_both_cameras():
    packet = synthetic_packet()
    results = [
        perturb(
            packet, {"family": "missing", "intensity": intensity}, evaluation_rng(21, 1)
        )
        for intensity in (0.3, 0.7, 0.9)
    ]
    for camera in CAMERAS:
        masks = [result[camera]["mask"] for result in results]
        assert not np.any(masks[1] & ~masks[0])
        assert not np.any(masks[2] & ~masks[1])
        assert masks[0].sum() > masks[1].sum() > masks[2].sum() > 0
        for result in results:
            assert not np.any(result[camera]["points"][~result[camera]["mask"]])


def test_depth_noise_preserves_camera_rays_colors_and_raw_rgb_depth():
    packet = synthetic_packet(padded=False)
    saved = copy.deepcopy(packet)
    sigma = 0.01
    result = perturb(
        packet, {"family": "depth", "intensity": sigma}, evaluation_rng(22, 4)
    )
    for camera in CAMERAS:
        transform = packet[camera]["base_from_camera"]
        before = (packet[camera]["points"][:, :3] - transform[:3, 3]) @ transform[
            :3, :3
        ]
        after = (result[camera]["points"][:, :3] - transform[:3, 3]) @ transform[:3, :3]
        assert result[camera]["mask"].all()
        # Axial noise changes depth along the calibrated ray, not image location.
        np.testing.assert_allclose(
            after[:, :2] / after[:, 2:3],
            before[:, :2] / before[:, 2:3],
            rtol=2e-6,
            atol=2e-7,
        )
        depth_delta = after[:, 2] - before[:, 2]
        assert abs(float(depth_delta.mean())) < 0.002
        assert 0.8 * sigma < float(depth_delta.std()) < 1.2 * sigma
        np.testing.assert_array_equal(
            result[camera]["points"][:, 3:], saved[camera]["points"][:, 3:]
        )
    assert_packet_unchanged(packet, saved)


def test_depth_intensities_share_noise_draws():
    packet = synthetic_packet(padded=False)
    low = perturb(packet, {"family": "depth", "intensity": 0.01}, evaluation_rng(23, 0))
    high = perturb(
        packet, {"family": "depth", "intensity": 0.02}, evaluation_rng(23, 0)
    )
    for camera in CAMERAS:
        original = packet[camera]["points"][:, :3]
        np.testing.assert_allclose(
            high[camera]["points"][:, :3] - original,
            2 * (low[camera]["points"][:, :3] - original),
            rtol=2e-5,
            atol=4e-7,
        )


def test_evaluation_rng_repeats_same_scene_step_and_separates_others():
    packet = synthetic_packet()
    condition = {"family": "missing", "intensity": 0.3}
    first = perturb(packet, condition, evaluation_rng(500000, 2))
    repeated = perturb(packet, condition, evaluation_rng(500000, 2))
    other_scene = perturb(packet, condition, evaluation_rng(500001, 2))
    other_step = perturb(packet, condition, evaluation_rng(500000, 3))
    assert packet_digest(first) == packet_digest(repeated)
    assert packet_digest(first) != packet_digest(other_scene)
    assert packet_digest(first) != packet_digest(other_step)


def test_generator_state_roundtrip_resumes_augmentation_and_corruption_exactly():
    packet = synthetic_packet()
    recipe = {
        "families": ["nominal", "absent", "occlusion", "depth", "missing"],
        "probabilities": [0.2] * 5,
        "occlusion_range": [0.1, 0.7],
        "depth_sigma_range_m": [0.003, 0.025],
        "missing_range": [0.1, 0.9],
    }
    original = np.random.default_rng(24)
    for _ in range(3):
        perturb(packet, augmentation_condition(original, recipe), original)
    # Exercise the serializable checkpoint payload, not just a shared Python object.
    saved_state = json.loads(json.dumps(original.bit_generator.state))
    resumed = np.random.default_rng(999)
    resumed.bit_generator.state = saved_state
    for _ in range(8):
        condition = augmentation_condition(original, recipe)
        resumed_condition = augmentation_condition(resumed, recipe)
        assert condition == resumed_condition
        expected = perturb(packet, condition, original)
        actual = perturb(packet, resumed_condition, resumed)
        assert packet_digest(expected) == packet_digest(actual)
        assert original.bit_generator.state == resumed.bit_generator.state
