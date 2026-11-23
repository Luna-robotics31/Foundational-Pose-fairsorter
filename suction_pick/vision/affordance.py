"""Single-cup suction affordance map from depth (stands in for the paper's SG-U-Net++).

The paper feeds the depth image to a trained network (SG-U-Net++) that outputs, per pixel,
whether ONE cup centred there can grasp; "pixels with non-zero grasp quality" form the
affordance map. That network and its weights are not published, so this module computes
the same kind of map from geometry. A pixel is graspable when a cup centred on it would
sit on a surface that is
  * complete: almost all depth pixels under the cup are valid (no holes, no edge),
  * flat: the RMS distance of those points from their best-fit plane is small
    (a depth step or strong curvature under the cup inflates it -> no seal),
  * not too steep, and
  * not the table/bin floor (height above the dominant support plane).
The patch is a square window of half-width = cup radius in pixels at the scene's median
distance, so it covers the whole cup disc (slightly conservative).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from suction_pick.settings.config import AffordanceParams


@dataclass
class SurfaceMap:
    points: np.ndarray        # (H, W, 3) planning frame, mm, NaN = invalid
    normals: np.ndarray       # (H, W, 3) unit, pointing "up" (+Z of the planning frame)
    rms: np.ndarray           # (H, W) plane-fit RMS under the cup, mm
    score: np.ndarray         # (H, W) 0 = not graspable, (0, 1] = graspable
    labels: np.ndarray        # (H, W) int32 graspable region id, 0 = none
    window_px: int
    table: tuple | None       # (unit normal, offset): n.p + offset = height above table
    stats: dict


def _box(img, k):
    return cv2.boxFilter(img, cv2.CV_64F, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)


def local_planes(points: np.ndarray, radius_px: int):
    """Per-pixel plane fit over a (2r+1)^2 window: normals, RMS residual, valid fraction."""
    valid = np.isfinite(points).all(axis=2)
    origin = np.nanmean(points[valid], axis=0)
    P = np.where(valid[..., None], points - origin, 0.0)
    k = 2 * radius_px + 1
    n = _box(valid.astype(np.float64), k)
    safe = np.maximum(n, 1.0)
    mean = np.dstack([_box(P[..., i], k) / safe for i in range(3)])
    cov = np.empty(points.shape[:2] + (3, 3))
    for i in range(3):
        for j in range(i, 3):
            c = _box(P[..., i] * P[..., j], k) / safe - mean[..., i] * mean[..., j]
            cov[..., i, j] = cov[..., j, i] = c
    cov[n < 3] = np.eye(3)
    vals, vecs = np.linalg.eigh(cov)
    normals = vecs[..., :, 0]
    normals *= np.where(normals[..., 2:3] < 0, -1.0, 1.0)  # point "up"
    rms = np.sqrt(np.maximum(vals[..., 0], 0.0))
    return normals, rms, n / (k * k), valid


def fit_table(points: np.ndarray, normals: np.ndarray, rng=None, iters=300, tol=4.0):
    """Largest near-horizontal plane = the table / bin floor the objects stand on.

    (A visible floor lower down, beside the table, is smaller in view and is ignored;
    an object top larger than the visible table would be mistaken for it.)

    Returns (unit normal, offset) so that height = normal.p + offset, or None.
    """
    rng = rng or np.random.default_rng(0)
    valid = np.isfinite(points).all(axis=2) & (normals[..., 2] > np.cos(np.radians(20)))
    pts = points[valid]
    if len(pts) < 500:
        return None
    pts = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
    best, best_count = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < np.cos(np.radians(20)):
            continue
        d = -n @ a
        count = int(np.sum(np.abs(pts @ n + d) < tol))
        if count > best_count:
            best, best_count = (n, d), count
    if best is None or best_count < 0.15 * len(pts):
        return None
    n, d = best
    inl = pts[np.abs(pts @ n + d) < tol]
    centre = inl.mean(axis=0)
    n = np.linalg.svd(inl - centre)[2][2]
    n = n if n[2] > 0 else -n
    return n, float(-n @ centre)


def compute_affordance(points: np.ndarray, distance_mm: np.ndarray, fx: float,
                       cup_radius: float, params: AffordanceParams, table=None, region=None) -> SurfaceMap:
    """points: (H, W, 3) in the planning frame (Z up, mm); distance_mm: camera depth (mm).
    table: (unit normal, offset) of the surface parcels stand on (e.g. the measured bin);
    default: the largest near-horizontal plane in view. region: (H, W) bool, pixels where parcels
    may be (e.g. inside the bin outline); default: everywhere."""
    if params.max_range > 0:  # working range: the floor / room behind the table never counts
        far = ~(np.nan_to_num(distance_mm) <= params.max_range)
        points = np.where(far[..., None], np.nan, points)
    valid_px = np.isfinite(points).all(axis=2)
    if valid_px.sum() < 1000:
        raise RuntimeError("Too few valid depth pixels.")
    median_dist = float(np.nanmedian(distance_mm[valid_px]))
    radius_px = max(2, int(round(cup_radius * fx / median_dist)))
    normals, rms, valid_frac, valid = local_planes(points, radius_px)

    flat_limit = np.maximum(params.flatness, params.noise_fraction * np.nan_to_num(distance_mm))
    tilt = np.degrees(np.arccos(np.clip(normals[..., 2], -1, 1)))
    complete = valid & (valid_frac >= params.min_valid_fraction)
    flat = rms < flat_limit
    upright = tilt < params.max_tilt
    if table is None and params.min_height > 0:
        table = fit_table(points, normals)
    if table is not None:
        height = np.nan_to_num(points @ table[0] + table[1], nan=-1e9)
        raised = height > params.min_height
    else:
        raised = np.ones_like(valid)
    ok = complete & flat & upright & raised
    if region is not None:
        ok &= region

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ok.astype(np.uint8), connectivity=8)
    for lab in range(1, n_labels):
        if stats[lab, cv2.CC_STAT_AREA] < params.min_region_px:
            labels[labels == lab] = 0
    # A graspable patch stops one cup radius before the parcel's edge. If it comes that
    # close to the image border, the parcel is cut off by the border: its visible part
    # would give a wrong centre, so it is not picked (move the camera / home pose instead).
    cut_off = []
    h_img, w_img = labels.shape
    margin = radius_px + 3
    for lab in np.unique(labels[labels > 0]):
        vs, us = np.nonzero(labels == lab)
        edges = [name for name, hit in (("left", us.min() < margin), ("right", us.max() >= w_img - margin),
                                        ("top", vs.min() < margin), ("bottom", vs.max() >= h_img - margin)) if hit]
        if edges:
            cut_off.append({"region": int(lab), "edges": edges, "pixel_centre": [int(us.mean()), int(vs.mean())]})
            labels[labels == lab] = 0
    ok &= labels > 0
    score = np.where(ok, np.clip(1.0 - rms / flat_limit, 0.05, 1.0) * np.cos(np.radians(tilt)), 0.0)
    counts = {
        "valid_px": int(valid.sum()), "complete": int(complete.sum()),
        "flat": int((complete & flat).sum()), "upright": int((complete & flat & upright).sum()),
        "graspable": int(ok.sum()), "regions": int(len(np.unique(labels[ok]))),
        "table_found": table is not None, "window_px": radius_px, "cut_off_at_image_edge": cut_off,
    }
    return SurfaceMap(points, normals, rms, score, labels.astype(np.int32), radius_px, table, counts)


def object_tilts(surface: SurfaceMap, level_tol_deg: float = 5.0) -> list[dict]:
    """Tilt of every graspable region's top surface relative to the table / bin floor.

    The reference "up" is the table plane's normal when one was found (so a camera or bin
    that is not level does not count as tilt), otherwise +Z of the planning frame.
    pitch = slope along the reference X axis, roll = slope along Y (deg): positive pitch means
    the surface rises towards +X (tan(pitch) mm per mm), positive roll rises towards +Y.
    tilt = total angle between the surface normal and "up".
    """
    up = surface.table[0] if surface.table is not None else np.array([0.0, 0.0, 1.0])
    # frame with Z = up and X = planning X projected into the table plane
    x = np.array([1.0, 0.0, 0.0]) - up[0] * up
    x /= np.linalg.norm(x)
    F = np.column_stack([x, np.cross(up, x), up])           # columns: reference axes
    out = []
    for lab in np.unique(surface.labels[surface.labels > 0]):
        pts = surface.points[surface.labels == lab]
        c = pts.mean(axis=0)
        n = np.linalg.svd(pts - c, full_matrices=False)[2][2]  # plane fit over the whole region
        n = n if n @ up > 0 else -n
        nl = F.T @ n                                          # normal in the reference frame
        tilt = float(np.degrees(np.arccos(np.clip(nl[2], -1, 1))))
        out.append({
            "region": int(lab), "centre_mm": np.round(c, 1).tolist(), "points": int(len(pts)),
            "tilt_deg": round(tilt, 2),
            "pitch_deg": round(float(np.degrees(np.arctan2(-nl[0], nl[2]))), 2),
            "roll_deg": round(float(np.degrees(np.arctan2(-nl[1], nl[2]))), 2),
            "tilted": tilt > level_tol_deg,
            "reference": "table plane" if surface.table is not None else "planning frame +Z",
        })
    return out
