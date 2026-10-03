"""Sensor-representation corruptions; no object poses or privileged masks."""

import hashlib
import numpy as np
from geopolicy.sensors import fuse_points

CAMERAS = ("agentview", "robot0_eye_in_hand")


def perturb(packet, condition, rng):
    """Always corrupt both named views, then choose model views downstream.

    Axial metric depth noise is applied along calibrated camera rays AFTER the
    preserved workspace crop/voxel/sample. Missingness is IID at sampled-point
    level. Occlusion is a centered square, with intensity its image-area fraction.
    RGB-D raw arrays are retained unmodified as provenance, never used after
    their point representation has been removed. Invalid points are finite zero.
    """
    family, value = condition["family"], float(condition["intensity"])
    assert family in ("nominal", "absent", "occlusion", "depth", "missing")
    assert value >= 0 and (family == "depth" or value <= 1)
    output = {}
    for camera in CAMERAS:
        original = packet[camera]
        p = original["points"].copy()
        m = original["mask"].copy()
        assert p.shape == (512, 6) and m.shape == (512,)
        assert np.isfinite(p).all()
        # Fixed-size draws couple intensity levels and keep views/seeds paired.
        noise = rng.standard_normal(512)
        uniform = rng.random(512)
        transform = original["base_from_camera"]
        k = original["intrinsic"]
        xyz_camera = (p[:, :3] - transform[:3, 3]) @ transform[:3, :3]
        z = xyz_camera[:, 2]
        valid_z = z > 1e-6
        if family == "absent" and camera == CAMERAS[0]:
            m[:] = False
        elif family == "occlusion" and camera == CAMERAS[0] and value:
            u = xyz_camera[:, 0] / np.where(valid_z, z, 1) * k[0, 0] + k[0, 2]
            v = xyz_camera[:, 1] / np.where(valid_z, z, 1) * k[1, 1] + k[1, 2]
            # All cameras in the preserved dataset are128x128.
            half = 64 * np.sqrt(value)
            blocked = (abs(u - 63.5) <= half) & (abs(v - 63.5) <= half) & valid_z
            m &= ~blocked
        elif family == "depth" and value:
            delta = (noise * value).astype(np.float32)
            ray = (p[:, :3] - transform[:3, 3]) / np.where(valid_z, z, 1)[:, None]
            p[:, :3] += ray * delta[:, None]
            m &= valid_z & (z + delta > 0) & (z + delta < 5)
        elif family == "missing":
            m &= uniform >= value
        p[~m] = 0
        output[camera] = dict(points=p, mask=m, intrinsic=k, base_from_camera=transform)
    return output


def fuse(packet, view):
    names = CAMERAS if view == "fusion" else CAMERAS[:1]
    assert view in ("fixed", "fusion")
    return fuse_points([(packet[c]["points"], packet[c]["mask"]) for c in names])


def packet_digest(packet, fields=("points", "mask")):
    h = hashlib.sha256()
    for camera in CAMERAS:
        for field in fields:
            a = np.ascontiguousarray(packet[camera][field])
            h.update(camera.encode())
            h.update(field.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def evaluation_rng(scene, step):
    return np.random.default_rng(np.random.SeedSequence([4127, scene, step]))


def augmentation_condition(rng, recipe):
    family = str(rng.choice(recipe["families"], p=recipe["probabilities"]))
    if family == "nominal":
        value = 0
    elif family == "absent":
        value = 1
    else:
        key = {
            "occlusion": "occlusion_range",
            "depth": "depth_sigma_range_m",
            "missing": "missing_range",
        }[family]
        value = float(rng.uniform(*recipe[key]))
    return dict(family=family, intensity=value)
