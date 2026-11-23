"""Generate a deterministic artificial scene with known segmentation masks."""
from pathlib import Path

import cv2
import numpy as np

from .artifacts import write_image


def make_scene():
    rng = np.random.default_rng(42)
    height, width = 480, 720
    background = np.full((height, width, 3), (115, 58, 23), np.uint8)
    background = np.clip(background.astype(np.int16) + rng.integers(-2, 3, background.shape), 0, 255).astype(np.uint8)
    color = background.copy()
    labels = np.zeros((height, width), np.uint16)
    # A carton, a white courier bag, and a grey courier bag with clear gaps.
    cv2.rectangle(labels, (65, 75), (245, 225), 1, -1)
    cv2.fillPoly(labels, [np.array([(345, 55), (580, 80), (610, 215),
                                   (565, 235), (340, 205), (325, 105)])], 2)
    cv2.ellipse(labels, (360, 365), (145, 65), -8, 0, 360, 3, -1)
    colors = [(100, 160, 205), (230, 230, 235), (145, 145, 150)]
    for index, shade in enumerate(colors, 1):
        color[labels == index] = shade
    cv2.line(color, (155, 77), (155, 223), (135, 185, 215), 12)
    cv2.rectangle(color, (390, 110), (495, 165), (245, 245, 245), -1)
    for x in range(398, 480, 5):
        cv2.line(color, (x, 118), (x, 155), (45, 45, 45), 2)
    cv2.line(color, (350, 180), (545, 215), (175, 175, 175), 2)
    cv2.line(color, (240, 365), (455, 350), (170, 170, 175), 3)
    background_depth = np.full((height, width), 900.0, np.float32)
    depth = background_depth.copy()
    for index, distance in enumerate((770.0, 850.0, 830.0), 1):
        depth[labels == index] = distance
    return color, background, depth, background_depth, labels


def write_demo_inputs(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    color, background, depth, background_depth, labels = make_scene()
    write_image(root / "scene.png", color)
    write_image(root / "background.png", background)
    write_image(root / "ground_truth_labels.png", labels)
    np.save(root / "depth_mm.npy", depth, allow_pickle=False)
    np.save(root / "background_depth_mm.npy", background_depth, allow_pickle=False)
    return color, background, depth, background_depth, labels
