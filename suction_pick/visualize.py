"""Overlay of graspable regions and planned cup placements on the colour image."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import project, transform

COLORS = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]


def draw_clearance(img, surface, grasp, K, T_cam_plan, gripper, params):
    """Obstacles near the chosen grasp's approach path (orange), points that would hit the
    gripper / camera (red), and the gripper volumes (outlines) at the cup plane."""
    from .planner import ee_hit_mask, ee_model
    ee = ee_model(gripper, params.hover_radius)
    P = surface.points
    ok = np.isfinite(P).all(axis=2)
    q = np.full(P.shape, np.nan)
    q[ok] = (P[ok] - grasp.tcp) @ grasp.R                       # gripper frame
    near = ok & (np.nan_to_num(q[..., 0]) ** 2 + np.nan_to_num(q[..., 1]) ** 2 < ee["clear_radius"] ** 2)
    obstacle = near & (np.nan_to_num(q[..., 2]) > 15.0)          # sticks up >15 mm above the parcel top
    hit = np.zeros(ok.shape, bool)
    hit[obstacle] = ee_hit_mask(q[obstacle], ee, params)
    img[obstacle] = (0.45 * img[obstacle] + [0, 90, 140]).astype(np.uint8)       # orange
    img[hit] = (0, 0, 255)                                                          # red
    # gripper volumes at the cup plane, projected into the image
    def poly(points_g, color, thick=1):
        cam = transform(T_cam_plan, grasp.tcp + points_g @ grasp.R.T)
        uv = np.round(project(K, cam)).astype(int).reshape(-1, 1, 2)
        cv2.polylines(img, [uv], True, color, thick, cv2.LINE_AA)
    hx, hy = ee["half"]
    poly(np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]]), (255, 255, 255), 1)
    t = np.linspace(0, 2 * np.pi, 48)
    for (cx, cy, r, _), color in zip(ee["columns"], [(255, 255, 0), (255, 0, 255)]):
        poly(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t), np.zeros_like(t)]), color, 2)
    return int(obstacle.sum()), int(hit.sum())


def overlay(bgr, surface, grasps, K, T_cam_plan, cup_radius_mm, tilts=(), top=5, gripper=None, params=None):
    img = bgr.copy()
    if img.shape[:2] != surface.labels.shape:
        img = cv2.resize(img, surface.labels.shape[::-1])
    if grasps and gripper is not None and params is not None:
        n_obs, n_hit = draw_clearance(img, surface, grasps[0], K, T_cam_plan, gripper, params)
        cv2.putText(img, f"orange: obstacles near path ({n_obs} px)  red: would hit ({n_hit} px)  "
                    "white: cups  cyan: tool body  magenta: camera", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    tint = np.zeros_like(img)
    rng = np.random.default_rng(1)
    for lab in np.unique(surface.labels[surface.labels > 0]):
        tint[surface.labels == lab] = rng.integers(60, 255, 3)
    mask = surface.labels > 0
    img[mask] = (0.55 * img[mask] + 0.45 * tint[mask]).astype(np.uint8)
    for t in tilts:  # object tilt label at the region centre
        u, v = np.round(project(K, transform(T_cam_plan, np.array([t["centre_mm"]])))[0]).astype(int)
        text = (f"TILT {t['tilt_deg']:.0f} (p{t['pitch_deg']:+.0f} r{t['roll_deg']:+.0f})"
                if t["tilted"] else f"level {t['tilt_deg']:.1f}")
        color = (0, 0, 255) if t["tilted"] else (255, 255, 255)
        cv2.putText(img, text, (u - 40, v + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (u - 40, v + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    for rank, g in reversed(list(enumerate(grasps[:top]))):
        color = COLORS[rank % len(COLORS)]
        cam = transform(T_cam_plan, np.vstack([g.cups, g.tcp]))
        uv = np.round(project(K, cam)).astype(int)
        r_px = [max(3, int(cup_radius_mm * K[0, 0] / c[2])) for c in cam[:-1]]
        cv2.polylines(img, [uv[:-1].reshape(-1, 1, 2)], False, color, 2)
        for (u, v), r, active in zip(uv[:-1], r_px, g.active):
            cv2.circle(img, (u, v), r, color, 2 if active else 1, cv2.LINE_AA)
            if not active:
                cv2.line(img, (u - r, v - r), (u + r, v + r), color, 1)
        u, v = uv[-1]
        cv2.drawMarker(img, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(img, f"#{rank + 1} {g.n_objects} obj", (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img


def affordance_image(surface):
    s = (np.clip(surface.score, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(s, cv2.COLORMAP_VIRIDIS) * (surface.score > 0)[..., None]
