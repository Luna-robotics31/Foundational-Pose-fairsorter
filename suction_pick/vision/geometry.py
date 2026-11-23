"""Depth image <-> 3D points and frame helpers (mm)."""

from __future__ import annotations

import numpy as np

# Camera optical frame (X right, Y down, Z forward) -> an "up" frame with Z pointing back
# towards the camera, used when no robot calibration is available.
T_UP_FROM_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def depth_to_points(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W) depth in mm -> (H, W, 3) points in the camera frame; invalid -> NaN."""
    h, w = depth_mm.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_mm.astype(np.float64)
    z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    return np.dstack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points (NaN stays NaN)."""
    return points @ T[:3, :3].T + T[:3, 3]


def project(K: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixel coordinates."""
    p = np.atleast_2d(points_cam)
    return np.column_stack((K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2],
                            K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]))


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors (broadcasting over the last axis)."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))
