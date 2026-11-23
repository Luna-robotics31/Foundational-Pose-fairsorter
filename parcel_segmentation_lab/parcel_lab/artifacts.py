"""Read inputs and save reproducible, non-overwriting segmentation runs."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .grasp_preview import draw_preview


def read_color(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read color image: {path}")
    return image


def read_depth(path):
    """Read .npy float/integer mm, or single-channel uint16 PNG in mm."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            raise ValueError("Depth PNG must be a single-channel uint16 image in mm. Use .npy for floating point depth.")
    if depth.ndim != 2:
        raise ValueError("Depth must be one channel, not a colorized depth visualization.")
    return depth.astype(np.float32)


def write_image(path, array):
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Cannot write image: {path}")


def draw_overlay(color, result, roi=None, target_instance_id=None):
    """Top-view instance masks with full-parcel boxes, like a sorting preview.

    Labels are per-frame parcel IDs, not a validated physical picking order.
    The optional target is chosen by the existing 2D cup-layout preview.
    """
    output = color.copy()
    palette = [(225, 90, 235), (100, 235, 90), (240, 145, 120),
               (210, 220, 70), (40, 185, 255), (110, 110, 255)]
    # Color masks first; draw all labels afterwards so masks never tint labels.
    for instance in result.instances:
        tint = np.array(palette[(instance.id - 1) % len(palette)], dtype=np.float32)
        selected = instance.mask > 0
        output[selected] = (output[selected] * 0.55 + tint * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(instance.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, tuple(int(v) for v in tint), 1)
    for instance in result.instances:
        x, y, w, h = instance.bbox_xywh
        active = instance.id == target_instance_id
        box_color = (220, 90, 125) if active else (0, 255, 0)
        text_color = (255, 255, 255) if active else (0, 0, 0)
        label = f"target {instance.id}" if active else f"parcel {instance.id}"
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), box_color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_w, label_h = text_w + 12, text_h + baseline + 8
        left = max(0, min(x, output.shape[1] - label_w))
        top = y - label_h if y - label_h >= 32 else max(32, y + 2)
        top = max(0, min(top, output.shape[0] - label_h))
        cv2.rectangle(output, (left, top), (left + label_w - 1, top + label_h - 1), box_color, -1)
        cv2.putText(output, label, (left + 6, top + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, text_color, 1, cv2.LINE_AA)
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), (0, 220, 255), 2)
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, 30), (25, 25, 25), -1)
    cv2.putText(output, f"TOP VIEW | {len(result.instances)} parcel candidates | IDs are not pick order",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def save_run(root, color, background, result, settings, depth_mm=None,
             background_depth_mm=None, metadata=None, prefix="run", grasp_preview=None):
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / f"{prefix}_{stamp}_{uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "masks").mkdir()
    (directory / "crops").mkdir()
    write_image(directory / "scene.png", color)
    write_image(directory / "background.png", background)
    overlay = draw_overlay(color, result, settings.roi)
    write_image(directory / "overlay.png", overlay)
    if grasp_preview is not None:
        target_overlay = draw_overlay(color, result, settings.roi, grasp_preview.target_instance_id)
        write_image(directory / "grasp_overlay.png", draw_preview(target_overlay, grasp_preview))
        (directory / "grasp_preview.json").write_text(json.dumps(asdict(grasp_preview), indent=2, allow_nan=False) + "\n")
    write_image(directory / "foreground.png", result.foreground)
    labels = np.zeros(color.shape[:2], np.uint16)
    records = []
    if len(result.instances) > np.iinfo(np.uint16).max:
        raise ValueError("Too many components for uint16 instance labels; increase minimum area.")
    for instance in result.instances:
        basename = f"parcel_{instance.id:03d}.png"
        write_image(directory / "masks" / basename, instance.mask)
        x, y, w, h = instance.bbox_xywh
        rgba = cv2.cvtColor(color[y:y + h, x:x + w], cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = instance.mask[y:y + h, x:x + w]
        write_image(directory / "crops" / basename, rgba)
        labels[instance.mask > 0] = instance.id
        records.append({
            "id": instance.id,
            "bbox_xywh_px": instance.bbox_xywh,
            "area_px": instance.area_px,
            "centroid_xy_px": instance.centroid_xy_px,
            "touches_roi_boundary": instance.touches_roi_boundary,
            "median_depth_mm": instance.median_depth_mm,
            "valid_depth_fraction": instance.valid_depth_fraction,
            "mask": f"masks/{basename}",
            "crop": f"crops/{basename}",
        })
    write_image(directory / "labels.png", labels)
    if depth_mm is not None:
        np.save(directory / "depth_mm.npy", depth_mm, allow_pickle=False)
        np.save(directory / "background_depth_mm.npy", background_depth_mm, allow_pickle=False)
    report = {
        "schema_version": 1,
        "method": "empty_reference_subtraction_and_connected_components",
        "count": len(records),
        "image_size_wh": [color.shape[1], color.shape[0]],
        "settings": asdict(settings),
        "instances": records,
        "warnings": result.warnings,
        "metadata": metadata or {},
        "grasp_preview_file": "grasp_preview.json" if grasp_preview is not None else None,
        "coordinate_note": "All xy/bbox/area measurements are in image pixels. Centroids are not grasp points. Depth is camera-axis distance in mm, not robot coordinates.",
    }
    (directory / "instances.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return directory
