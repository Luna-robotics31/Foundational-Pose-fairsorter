"""Gemini 2 capture through the cube_pick tracker's camera code (V4L2, Y16 depth, rectified)."""

from __future__ import annotations

import numpy as np


def capture(n_frames: int = 5, warmup: int = 10):
    """Return (bgr, depth_mm, K): per-pixel median depth of n_frames (reduces noise).

    Depth is aligned to the colour image and both are undistorted, so K is a plain pinhole.
    """
    from cube_pick.vision.live_tracker import OrbbecCamera  # pyorbbecsdk only needed here

    camera = OrbbecCamera()
    try:
        frames = []
        for _ in range(warmup + n_frames * 4):
            out = camera.read()
            if out is None:
                continue
            frames.append(out)
            if len(frames) >= warmup + n_frames:
                break
        frames = frames[warmup:]
        if len(frames) < n_frames:
            raise RuntimeError(f"Only {len(frames)} camera frames received; check the camera.")
        depth = np.stack([d for _, d, _ in frames]) * 1000.0  # m -> mm
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        depth = np.nan_to_num(np.nanmedian(depth, axis=0), nan=0.0).astype(np.float32)
        return frames[-1][0], depth, camera.K.copy()
    finally:
        camera.stop()
