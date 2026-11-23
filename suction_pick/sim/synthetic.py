"""Synthetic top-down depth scene with known answers, for testing without camera or robot.

Camera 600 mm above a table, looking straight down (640x480, fx = fy = 600).
Planning frame = camera "up" frame: X right, Y up in the image, Z towards the camera (mm).
"""

from __future__ import annotations

import numpy as np

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
TABLE_Z = -600.0

# name, centre (x, y), size (x, y), height above table, tilt: deg or (slope along X, slope along Y)
BOXES = [
    ("small_A", (-35.0, 60.0), (50.0, 50.0), 80.0, 0.0),   # A and B: 70 mm apart, 20 mm gap,
    ("small_B", (35.0, 60.0), (50.0, 50.0), 80.0, 0.0),    #   too small for two cups alone
    ("large_C", (-110.0, -80.0), (150.0, 90.0), 50.0, 0.0),  # long box: both cups on it
    ("tilted_D", (110.0, -80.0), (120.0, 80.0), 60.0, 20.0),  # tilted top, 20 deg
    ("tiny_E", (150.0, 70.0), (25.0, 25.0), 40.0, 0.0),     # smaller than one cup
]


def render(noise_mm=0.5, seed=0, boxes=None):
    """Z-buffer a fine height field into the camera. Returns (depth_mm, truth dict)."""
    boxes = BOXES if boxes is None else boxes
    step = 0.4
    xs = np.arange(-260, 260, step)
    ys = np.arange(-200, 200, step)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full(X.shape, TABLE_Z)
    owner = np.full(X.shape, -1)
    for k, (_, (cx, cy), (sx, sy), h, tilt) in enumerate(boxes):
        inside = (np.abs(X - cx) <= sx / 2) & (np.abs(Y - cy) <= sy / 2)
        tx, ty = (tilt, 0.0) if np.isscalar(tilt) else tilt  # slope along X (pitch), along Y (roll), deg
        top = TABLE_Z + h + np.tan(np.radians(tx)) * (X - cx) + np.tan(np.radians(ty)) * (Y - cy)
        Z = np.where(inside, top, Z)
        owner = np.where(inside, k, owner)
    # camera frame: x = X, y = -Y, z = -Z
    zc = -Z
    u = np.round(K[0, 0] * X / zc + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * (-Y) / zc + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    depth = np.full((480, 640), np.inf)
    np.minimum.at(depth, (v[ok], u[ok]), zc[ok])
    depth[~np.isfinite(depth)] = 0
    rng = np.random.default_rng(seed)
    depth = np.where(depth > 0, depth + rng.normal(0, noise_mm, depth.shape), 0)
    truth = {name: {"centre": c, "size": s, "top_z": TABLE_Z + h, "tilt": t}
             for name, c, s, h, t in boxes}
    return depth.astype(np.float32), truth


def box_under(point_xy, margin=0.0, boxes=None):
    """Name of the box whose top contains this (x, y), or None."""
    for name, (cx, cy), (sx, sy), _, _ in (BOXES if boxes is None else boxes):
        if abs(point_xy[0] - cx) <= sx / 2 - margin and abs(point_xy[1] - cy) <= sy / 2 - margin:
            return name
    return None


def true_normal(tilt):
    """Unit normal (pointing up) of a top surface with this tilt."""
    tx, ty = (tilt, 0.0) if np.isscalar(tilt) else tilt
    n = np.array([-np.tan(np.radians(tx)), -np.tan(np.radians(ty)), 1.0])
    return n / np.linalg.norm(n)
