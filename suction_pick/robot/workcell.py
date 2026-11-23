"""Virtual bin: the board (cubicle floor) outline with walls of a chosen height.

`python run.py bin` measures the board once from the camera: the largest plane in view is the
board; its points give a rectangle (minimum-area box in the board plane). The 4 corners are
saved in ROBOT BASE coordinates (bin.json). At every pick, walls are generated along the 4 edges
from the board surface up to --bin_height (default 100 mm) and added to the obstacle points, so
the end-effector collision check keeps the cups, tool body, wrist and camera out of them and the
hover point goes above the rim.
"""

from __future__ import annotations

import json
from datetime import datetime

import cv2
import numpy as np

from suction_pick.settings.paths import BIN_FILE  # data/bin.json


def find_planes(points_cam, max_range, tol, max_planes=4, min_share=0.08):
    """Large planes in view, biggest first: [(normal towards camera, offset c, share of points, rms)]."""
    from suction_pick.robot.calibrate import fit_board_plane
    P = points_cam.copy()
    finite = np.isfinite(points_cam).all(axis=2) & (np.nan_to_num(points_cam[..., 2], nan=1e9) <= max_range)
    total = max(int(finite.sum()), 1)
    planes = []
    for _ in range(max_planes):
        try:
            n, c, _, rms = fit_board_plane(P, max_range, tol)
        except RuntimeError:
            break
        inl = np.isfinite(P).all(axis=2) & (np.abs(np.nan_to_num(P @ n) - c) < tol)
        share = inl.sum() / total
        if share < min_share:
            break
        planes.append((n, c, float(share), rms))
        P[inl] = np.nan
    return planes


def measure_board(points_cam: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                  max_range: float = 2500.0, tol: float = 6.0, edge_px: int = 6):
    """Rectangle of the table / board: the NEAREST large horizontal plane in view (so a floor
    visible around a low table is not taken), plus the floor height below it if seen.

    Returns dict: corners_base (4, 3), normal_base (up), size_mm (long, short), cut_sides
    (rectangle sides that run outside the image, i.e. not fully seen), floor_z, inlier fraction.
    """
    planes = find_planes(points_cam, max_range, tol)
    level = []
    for n_, c_, share, rms_ in planes:               # keep planes that are horizontal in the robot frame
        nb_ = T_base_cam[:3, :3] @ n_
        if abs(nb_[2]) > np.cos(np.radians(30)):
            level.append((abs(c_), n_, c_, share, rms_))
    if not level:
        raise RuntimeError("No large horizontal surface in view.")
    level.sort(key=lambda t: t[0])                   # nearest to the camera first
    _, n, c, frac, rms = level[0]
    floor_z = None
    for _, n2, c2, _, _ in level[1:]:                # a lower plane further away = the floor
        p_board = T_base_cam[:3, :3] @ (c * n) + T_base_cam[:3, 3]
        p_other = T_base_cam[:3, :3] @ (c2 * n2) + T_base_cam[:3, 3]
        if p_other[2] < p_board[2] - 50:
            floor_z = float(p_other[2]) if floor_z is None else min(floor_z, float(p_other[2]))
    on = np.isfinite(points_cam).all(axis=2)
    on &= np.abs(np.nan_to_num(points_cam @ n) - c) < tol
    on &= np.nan_to_num(points_cam[..., 2], nan=1e9) <= max_range
    mask = cv2.morphologyEx(on.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if k < 2:
        raise RuntimeError("No board found in view.")
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    P = points_cam[(lab == big) & on]                         # board points, camera frame
    # 2D coordinates in the board plane
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    o = c * n                                                 # point of the plane
    uv = np.column_stack([(P - o) @ e1, (P - o) @ e2]).astype(np.float32)
    # Orientation from the minimum-area rectangle; extent from quantiles along its axes, so a few
    # stray "board" pixels at the noisy depth edges do not push the walls outward.
    _, _, ang = cv2.minAreaRect(uv)
    a = np.radians(ang)
    ax1, ax2 = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    s1, s2 = uv @ ax1, uv @ ax2
    lo1, hi1 = np.percentile(s1, [0.3, 99.7])
    lo2, hi2 = np.percentile(s2, [0.3, 99.7])
    w, h = hi1 - lo1, hi2 - lo2
    box = np.array([lo1 * ax1 + lo2 * ax2, hi1 * ax1 + lo2 * ax2, hi1 * ax1 + hi2 * ax2, lo1 * ax1 + hi2 * ax2])
    corners_cam = np.array([o + a * e1 + b * e2 for a, b in box])
    # sides that leave the image were not seen completely
    cut = []
    H, W = points_cam.shape[:2]
    for i in range(4):
        a, b = corners_cam[i], corners_cam[(i + 1) % 4]
        s = a + np.linspace(0, 1, 60)[:, None] * (b - a)
        u = K[0, 0] * s[:, 0] / s[:, 2] + K[0, 2]
        v = K[1, 1] * s[:, 1] / s[:, 2] + K[1, 2]
        inside = (s[:, 2] > 0) & (u > edge_px) & (u < W - edge_px) & (v > edge_px) & (v < H - edge_px)
        if inside.mean() < 0.9:
            cut.append(i)
    corners_base = (T_base_cam[:3, :3] @ corners_cam.T).T + T_base_cam[:3, 3]
    nb = T_base_cam[:3, :3] @ n
    nb = -nb if nb[2] < 0 else nb                             # up
    return {"corners_base": corners_base, "normal_base": nb, "size_mm": sorted([float(w), float(h)], reverse=True),
            "cut_sides": cut, "board_fraction": float(frac), "flatness_mm": float(rms), "floor_z": floor_z,
            "planes_found": len(planes)}


def save(board, path=BIN_FILE):
    data = {"corners_base_mm": np.round(board["corners_base"], 2).tolist(),
            "normal_base": np.round(board["normal_base"], 5).tolist(),
            "size_mm": [round(v, 1) for v in board["size_mm"]],
            "cut_sides": board["cut_sides"],
            "floor_z_mm": None if board.get("floor_z") is None else round(board["floor_z"], 1),
            "date": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def load(path=BIN_FILE):
    try:
        d = json.loads(path.read_text())
        return {"corners_base": np.array(d["corners_base_mm"], float), "normal_base": np.array(d["normal_base"], float),
                "size_mm": d["size_mm"], "date": d.get("date", "?"), "cut_sides": d.get("cut_sides", []),
                "floor_z": d.get("floor_z_mm")}
    except (OSError, KeyError, ValueError):
        return None


def wall_points(bin_, height: float, step: float = 5.0, bottom: float = 0.0) -> np.ndarray:
    """Points on the 4 virtual walls: along each board edge, from `bottom` mm (negative = below the
    board, e.g. the sides of the stand) up to `height` mm above the board."""
    C, n = bin_["corners_base"], bin_["normal_base"]
    hs = np.arange(bottom, height + 1e-6, step)
    out = []
    for i in range(4):
        a, b = C[i], C[(i + 1) % 4]
        ts = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step) + 1))
        edge = a + ts[:, None] * (b - a)
        out.append((edge[:, None, :] + hs[None, :, None] * n).reshape(-1, 3))
    return np.vstack(out)


def arm_wall_points(bin_, height: float, step: float = 10.0) -> np.ndarray:
    """Walls for the WHOLE ARM: the table sides from the floor (or the robot base plane) up to the
    board, plus the virtual walls `height` mm above it."""
    board_z = float(np.mean(bin_["corners_base"][:, 2]))
    # down to the floor under the table if it was seen, else to the robot base plane (z = 0)
    floor = bin_.get("floor_z")
    low = floor if floor is not None else min(0.0, board_z)
    return wall_points(bin_, height, step, bottom=-max(board_z - low, 0.0))


def draw(img, bin_, height, K, T_cam_base):
    """Board edges (yellow) and wall tops (orange) on the image."""
    C = bin_["corners_base"]
    top = C + height * bin_["normal_base"]
    for pts, color in ((C, (0, 255, 255)), (top, (0, 140, 255))):
        cam = (T_cam_base[:3, :3] @ pts.T).T + T_cam_base[:3, 3]
        if (cam[:, 2] <= 0).any():
            continue
        uv = np.column_stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2], K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]])
        cv2.polylines(img, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"bin walls {height:.0f} mm (yellow: board edge, orange: wall top)", (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)


def inside_mask(bin_, points):
    """(H, W) bool: points (robot base frame) whose XY lies inside the bin outline."""
    C = bin_["corners_base"][:, :2]
    P = np.nan_to_num(points[..., :2], nan=1e9)
    inside = np.ones(P.shape[:2], bool)
    sign = np.sign(np.cross(C[1] - C[0], C[2] - C[1]))            # orientation of the outline
    for i in range(4):
        a, b = C[i], C[(i + 1) % 4]
        cross = (b[0] - a[0]) * (P[..., 1] - a[1]) - (b[1] - a[1]) * (P[..., 0] - a[0])
        inside &= np.sign(cross) * sign >= 0
    return inside & np.isfinite(points).all(axis=2)
