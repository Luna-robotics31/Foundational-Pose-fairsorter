"""Virtual bin: the board (cubicle floor) outline with walls of a chosen height.

`python run.py bin` measures the board once from the camera: every large horizontal plane is a
candidate, each gives a rectangle (minimum-area box in its plane), and the largest one fully in
view is proposed. In the window (app/bin_ui.py) the user accepts it, picks another one, or clicks
the 4 corners by hand. The 4 corners are saved in ROBOT BASE coordinates (bin.json). At every
pick, walls are generated along the 4 edges
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


def _rectangle(points_cam, n, c, K, tol, max_range, edge_px):
    """Rectangle of one plane's points: (corners in the camera frame (4, 3), length, width, cut sides).
    Cut sides = sides that run outside the image (that edge was not seen)."""
    on = np.isfinite(points_cam).all(axis=2)
    on &= np.abs(np.nan_to_num(points_cam @ n) - c) < tol
    on &= np.nan_to_num(points_cam[..., 2], nan=1e9) <= max_range
    mask = cv2.morphologyEx(on.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if k < 2:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    P = points_cam[(lab == big) & on]                         # plane points, camera frame
    if len(P) < 500:
        return None
    e1 = np.cross(n, [0.0, 1.0, 0.0])                         # 2D coordinates in the plane
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    o = c * n
    uv = np.column_stack([(P - o) @ e1, (P - o) @ e2]).astype(np.float32)
    # Orientation from the minimum-area rectangle; extent from quantiles along its axes, so a few
    # stray pixels at the noisy depth edges do not push the walls outward.
    _, _, ang = cv2.minAreaRect(uv)
    t = np.radians(ang)
    ax1, ax2 = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    s1, s2 = uv @ ax1, uv @ ax2
    lo1, hi1 = np.percentile(s1, [0.3, 99.7])
    lo2, hi2 = np.percentile(s2, [0.3, 99.7])
    box = np.array([lo1 * ax1 + lo2 * ax2, hi1 * ax1 + lo2 * ax2, hi1 * ax1 + hi2 * ax2, lo1 * ax1 + hi2 * ax2])
    corners = np.array([o + x * e1 + y * e2 for x, y in box])
    cut = []
    H, W = points_cam.shape[:2]
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        q = p0 + np.linspace(0, 1, 60)[:, None] * (p1 - p0)
        u = K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]
        v = K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]
        inside = (q[:, 2] > 0) & (u > edge_px) & (u < W - edge_px) & (v > edge_px) & (v < H - edge_px)
        if inside.mean() < 0.9:
            cut.append(i)
    return corners, float(hi1 - lo1), float(hi2 - lo2), cut


def find_surfaces(points_cam: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                  max_range: float = 2500.0, tol: float = 6.0, edge_px: int = 6, verbose: bool = True):
    """All large horizontal surfaces in view, and which one is automatically taken as the table.

    Returns (candidates, auto_index). Each candidate: n, c (plane n.p = c, camera frame), share
    of the view, rms, rect = (corners_cam (4, 3), length, width, cut sides), z (robot base).
    Automatic choice = the LARGEST surface whose outline is fully inside the image (the floor
    runs out of the image; a box or parcel on the table is smaller). If none is fully in view:
    the nearest surface covering >= 20 % of the view.
    """
    cands = []
    for n_, c_, share, rms_ in find_planes(points_cam, max_range, tol):
        nb_ = T_base_cam[:3, :3] @ n_
        if abs(nb_[2]) <= np.cos(np.radians(30)):              # keep horizontal surfaces only
            continue
        rect = _rectangle(points_cam, n_, c_, K, tol, max_range, edge_px)
        if rect is None:
            continue
        z_ = float((T_base_cam[:3, :3] @ (c_ * n_) + T_base_cam[:3, 3])[2])
        cands.append({"n": n_, "c": c_, "share": share, "rms": rms_, "rect": rect, "z": z_})
    if not cands:
        raise RuntimeError("No large horizontal surface in view.")
    complete = [i for i, t in enumerate(cands) if not t["rect"][3]]
    if complete:
        auto = max(complete, key=lambda i: cands[i]["rect"][1] * cands[i]["rect"][2])
    else:
        big = [i for i, t in enumerate(cands) if t["share"] >= 0.20] or list(range(len(cands)))
        auto = min(big, key=lambda i: abs(cands[i]["c"]))
    if verbose:
        for i, t in enumerate(cands):
            print(f"[BIN] surface {i + 1}: {t['rect'][1]:.0f} x {t['rect'][2]:.0f} mm, {t['share']:.0%} of the view, "
                  f"z {t['z']:.0f} mm, {'fully in view' if not t['rect'][3] else 'runs out of the image'}"
                  f"{'   <- automatic choice' if i == auto else ''}")
    return cands, auto


def _floor_below(cands, table_z):
    lower = [t["z"] for t in cands if t["z"] < table_z - 50]   # >= 50 mm below the table = the floor
    return float(min(lower)) if lower else None


def _board(corners_cam, n_cam, T_base_cam, cut, frac, rms, floor_z, source):
    corners_base = (T_base_cam[:3, :3] @ corners_cam.T).T + T_base_cam[:3, 3]
    nb = T_base_cam[:3, :3] @ n_cam
    nb = -nb if nb[2] < 0 else nb                             # up
    sides = [float(np.linalg.norm(corners_base[(i + 1) % 4] - corners_base[i])) for i in range(4)]
    size = sorted([(sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2], reverse=True)
    return {"corners_base": corners_base, "normal_base": nb, "size_mm": size, "cut_sides": cut,
            "board_fraction": float(frac), "flatness_mm": float(rms), "floor_z": floor_z, "source": source}


def board_from_surface(cands, index, T_base_cam):
    """The bin = the rectangle of detected surface `index`."""
    t = cands[index]
    corners_cam, _, _, cut = t["rect"]
    return _board(corners_cam, t["n"], T_base_cam, cut, t["share"], t["rms"], _floor_below(cands, t["z"]),
                  f"detected surface {index + 1}")


def board_from_pixels(pixels, points_cam, K, T_base_cam, cands=(), max_range=2500.0, tol=6.0):
    """The bin = the 4 image points the user clicked.

    The table surface is the largest plane among the depth points INSIDE the clicked outline (a
    parcel on the table is smaller than the table), and each click becomes the 3D point where its
    viewing ray meets that surface.
    """
    from suction_pick.robot.calibrate import fit_board_plane
    px = np.asarray(pixels, float)
    centre = px.mean(axis=0)
    px = px[np.argsort(np.arctan2(px[:, 1] - centre[1], px[:, 0] - centre[0]))]   # convex order
    mask = np.zeros(points_cam.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
    P = np.where(mask[..., None] > 0, points_cam, np.nan)
    n, c, frac, rms = fit_board_plane(P, max_range, tol)
    rays = np.column_stack([(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)])
    corners_cam = rays * (c / (rays @ n))[:, None]
    z = float((T_base_cam[:3, :3] @ (c * n) + T_base_cam[:3, 3])[2])
    return _board(corners_cam, n, T_base_cam, [], frac, rms, _floor_below(cands, z), "selected by hand (4 clicks)")


def measure_board(points_cam: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                  max_range: float = 2500.0, tol: float = 6.0, edge_px: int = 6):
    """Automatic bin (no window): the surface find_surfaces() picks."""
    cands, auto = find_surfaces(points_cam, K, T_base_cam, max_range, tol, edge_px)
    return board_from_surface(cands, auto, T_base_cam)


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
import rich

def inside_mask(bin_, points):
    """(H, W) bool: points (robot base frame) whose XY lies inside the bin outline."""
    C = bin_["corners_base"][:, :2]
    P = np.nan_to_num(points[..., :2], nan=1e9)
    inside = np.ones(P.shape[:2], bool)
    rich.print("points",C)
    u, v = C[1] - C[0], C[2] - C[1]
    sign = np.sign(u[0] * v[1] - u[1] * v[0])  # z of the 2D cross product (np.cross needs 3D)
    rich.print("sign",sign)          # orientation of the outline
    for i in range(4):
        a, b = C[i], C[(i + 1) % 4]
        cross = (b[0] - a[0]) * (P[..., 1] - a[1]) - (b[1] - a[1]) * (P[..., 0] - a[0])
        inside &= np.sign(cross) * sign >= 0
    return inside & np.isfinite(points).all(axis=2)
