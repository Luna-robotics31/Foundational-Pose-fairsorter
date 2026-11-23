"""Empty-scene subtraction followed by connected-component instance masks.

This finds changed objects, not semantic parcel classes. Each separated foreground
component is a candidate parcel. Pixel centres below are measurements, not grasps.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Settings:
    mode: str = "rgb"
    color_threshold: float = 25.0
    min_depth_change_mm: float = 15.0
    min_area: int = 1000
    open_kernel: int = 3
    roi: Optional[Tuple[int, int, int, int]] = None

    def validate(self):
        if self.mode not in ("rgb", "depth", "hybrid"):
            raise ValueError("Mode must be rgb, depth, or hybrid.")
        if not np.isfinite(self.color_threshold) or self.color_threshold <= 0:
            raise ValueError("Color threshold must be a positive finite number.")
        if not np.isfinite(self.min_depth_change_mm) or self.min_depth_change_mm <= 0:
            raise ValueError("Depth change must be a positive finite number in mm.")
        if self.min_area < 1:
            raise ValueError("Minimum area must be at least 1 pixel.")
        if self.open_kernel < 0 or (self.open_kernel > 0 and self.open_kernel % 2 == 0):
            raise ValueError("Opening kernel must be 0 (disabled) or a positive odd number.")


@dataclass
class Instance:
    id: int
    mask: np.ndarray
    bbox_xywh: Tuple[int, int, int, int]
    area_px: int
    centroid_xy_px: Tuple[float, float]
    touches_roi_boundary: bool
    median_depth_mm: Optional[float]
    valid_depth_fraction: Optional[float]


@dataclass
class Result:
    foreground: np.ndarray
    instances: List[Instance]
    warnings: List[str]


def validate_color(image: np.ndarray):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Color images must be uint8 BGR images with three channels.")


def validate_depth(depth: np.ndarray, shape):
    if depth.ndim != 2 or depth.shape != shape:
        raise ValueError("Depth must be aligned to color, with exactly the same height and width; do not resize it to simulate alignment.")
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError("Depth must contain numeric distances in millimetres.")


def roi_mask(shape, roi):
    height, width = shape
    x, y, w, h = roi if roi is not None else (0, 0, width, height)
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("ROI must be x y width height, entirely inside the image.")
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return mask, (x, y, w, h)


def segment(color, background, settings, depth_mm=None, background_depth_mm=None):
    """Return one binary mask per connected foreground component.

    Depth foreground means closer to the camera than the empty reference. This
    is a camera-axis distance difference, not height above a tilted bin floor.
    Hybrid uses depth where both measurements are valid; RGB is a fallback only
    at invalid depth pixels. Missing depth is never treated as a nearer surface.
    """
    settings.validate()
    validate_color(color)
    validate_color(background)
    if color.shape != background.shape:
        raise ValueError("Scene and background must have identical resolution and camera pose.")
    shape = color.shape[:2]
    region, (rx, ry, rw, rh) = roi_mask(shape, settings.roi)
    warnings = []
    if (depth_mm is None) != (background_depth_mm is None):
        raise ValueError("Supply both scene depth and background depth, or neither.")
    if depth_mm is not None:
        validate_depth(depth_mm, shape)
        validate_depth(background_depth_mm, shape)
    if settings.mode != "rgb" and depth_mm is None:
        raise ValueError("Depth/hybrid mode requires aligned scene and background depth in mm.")

    # Small blur suppresses sensor noise. A small gap can still be lost; keep
    # parcels visibly separated and tune the threshold with the saved masks.
    scene_lab = cv2.cvtColor(cv2.GaussianBlur(color, (3, 3), 0), cv2.COLOR_BGR2LAB)
    background_lab = cv2.cvtColor(cv2.GaussianBlur(background, (3, 3), 0), cv2.COLOR_BGR2LAB)
    delta = scene_lab.astype(np.float32) - background_lab.astype(np.float32)
    color_changed = np.linalg.norm(delta, axis=2) >= settings.color_threshold
    foreground = color_changed
    if settings.mode != "rgb":
        current = depth_mm.astype(np.float32)
        reference = background_depth_mm.astype(np.float32)
        valid = np.isfinite(current) & np.isfinite(reference) & (current > 0) & (reference > 0)
        closer = valid & ((reference - current) >= settings.min_depth_change_mm)
        foreground = closer if settings.mode == "depth" else closer | (~valid & color_changed)
        invalid_fraction = float(np.mean(~valid[region > 0]))
        if invalid_fraction > 0.1:
            warnings.append(f"{invalid_fraction:.0%} of the ROI lacks a valid depth pair; masks may be incomplete or rely on RGB fallback.")

    binary = np.where(foreground & (region > 0), 255, 0).astype(np.uint8)
    if settings.open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings.open_kernel,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # No dilation/closing or convex hull: those can bridge separate parcels.
    binary[region == 0] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= settings.min_area]
    kept.sort(key=lambda label: (stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_LEFT]))
    instances = []
    cleaned = np.zeros(shape, np.uint8)
    for instance_id, label in enumerate(kept, 1):
        mask = np.where(labels == label, 255, 0).astype(np.uint8)
        cleaned[mask > 0] = 255
        x, y, w, h, area = (int(v) for v in stats[label])
        touches = x == rx or y == ry or x + w == rx + rw or y + h == ry + rh
        median_depth, valid_fraction = None, None
        if depth_mm is not None:
            values = depth_mm[mask > 0]
            valid_values = values[np.isfinite(values) & (values > 0)]
            valid_fraction = len(valid_values) / len(values)
            if len(valid_values):
                median_depth = float(np.median(valid_values))
        instances.append(Instance(instance_id, mask, (x, y, w, h), area,
                                  tuple(float(v) for v in centroids[label]), touches,
                                  median_depth, valid_fraction))
    if any(item.touches_roi_boundary for item in instances):
        warnings.append("A component touches the ROI boundary and may be clipped or include a bin wall.")
    if np.count_nonzero(cleaned) / np.count_nonzero(region) > 0.8:
        warnings.append("Foreground covers over 80% of the ROI. Check camera pose, lighting, background, and threshold.")
    return Result(cleaned, instances, warnings)
