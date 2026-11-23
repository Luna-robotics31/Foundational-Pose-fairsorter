"""Camera (hand-eye) calibration from the board and one fixed object on it, no markers needed.

The wrist camera looks at a flat board with one fixed object on it (e.g. the foam disc) from
several robot poses that tilt the flange in different directions around that object.

ROTATION (flange -> camera, R_fc): the board's normal n_i is measured in the camera frame at
every pose (plane fit). A correct R_fc maps all of them onto ONE normal n_b in the base frame:

    R_base_flange_i · R_fc · n_i = n_b                       (least squares; board need not be level)

POSITION (flange -> camera, t_fc): the object's top-surface centre x_i (camera frame) and the
board offset c_i (n_i·p = c_i on the board) must also map to ONE point p and ONE height h:

    R_bf_i (R_fc x_i + t_fc) + t_bf_i = p                    (3 equations per pose)
    c_i + (R_bf_i^T n_b)·t_fc + n_b·t_bf_i = h               (1 equation per pose)

which is linear in (t_fc, p, h). Tilting about different axes makes everything observable.

CHECK: leave-one-out. For every pose, solve with the OTHER poses and predict where that pose
sees the object; the miss in mm is an honest accuracy figure (the pose was not used to fit).
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from suction_pick.settings.paths import CALIB_FILE  # data/camera_calibration.json


# ── measurements (camera frame, mm) ─────────────────────────────────────
def fit_board_plane(points_cam: np.ndarray, max_range: float = 1200.0, tol: float = 4.0, rng=None):
    """Largest plane in camera-frame points (H, W, 3) mm. Returns (normal towards the camera,
    signed offset c = n.p of the plane, inlier fraction, RMS residual mm)."""
    rng = rng or np.random.default_rng(0)
    P = points_cam[np.isfinite(points_cam).all(axis=2)]
    P = P[P[:, 2] <= max_range]
    if len(P) < 2000:
        raise RuntimeError("Too few depth points on the board.")
    S = P[rng.choice(len(P), min(len(P), 30000), replace=False)]
    best = (0, None, None)
    for _ in range(400):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        count = int(np.sum(np.abs((S - a) @ n) < tol))
        if count > best[0]:
            best = (count, n, a)
    _, n, a = best
    inl = P[np.abs((P - a) @ n) < tol]
    centre = inl.mean(axis=0)
    n = np.linalg.svd(inl - centre, full_matrices=False)[2][2]
    if n[2] > 0:            # camera looks along +Z, so the board's normal towards it has n_z < 0
        n = -n
    rms = float(np.sqrt(np.mean(((inl - centre) @ n) ** 2)))
    return n, float(n @ centre), len(inl) / len(P), rms


def find_landmark(points_cam: np.ndarray, n, c, max_range=1200.0, min_height=15.0,
                  max_height=250.0, min_area=1500, edge=5):
    """Centre of the top surface of the largest object on the board that is fully in view.

    Returns (centre (3,) camera frame mm, info dict) or (None, reason). The top surface is the
    part within 6 mm of the object's top, so side walls seen at an angle do not shift it.
    """
    h = points_cam @ n - c                              # height above the board (mm)
    z = points_cam[..., 2]
    mask = (np.nan_to_num(h, nan=-1) > min_height) & (np.nan_to_num(h, nan=1e9) < max_height) \
        & (np.nan_to_num(z, nan=1e9) < max_range)
    k, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    H, W = mask.shape
    for i in np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1:
        x, y, w, hh, area = st[i]
        if area < min_area:
            break
        if x < edge or y < edge or x + w > W - edge or y + hh > H - edge:
            continue                                     # cut off by the image border
        hs = h[lab == i]
        top = np.percentile(hs, 95)
        sel = (lab == i) & (h > top - 6.0)
        pts = points_cam[sel]
        # Area-weighted centre: a pixel covers z^2 / (f^2 cos(view angle)) of the surface, so the
        # near side of the object has more pixels. A plain pixel mean shifts the centre towards
        # the camera (the same way in every pose -> it would bias the camera position).
        dist = np.linalg.norm(pts, axis=1)
        cos_view = np.clip(np.abs(pts @ n) / dist, 0.2, 1.0)
        wt = pts[:, 2] ** 2 / cos_view
        return (pts * wt[:, None]).sum(axis=0) / wt.sum(), {"area_px": int(area), "top_mm": round(float(top), 1),
                                  "top_px": int(sel.sum()), "pixel": [int(x + w / 2), int(y + hh / 2)]}
    return None, "no object fully in view"


# ── solvers ─────────────────────────────────────────────────────────────
def solve_rotation(R_base_flange: list, normals_cam: list, R_fc_init: np.ndarray):
    """Least-squares R_fc and board normal n_b from per-pose flange rotations and normals."""
    nb0 = np.mean([Rb @ R_fc_init @ n for Rb, n in zip(R_base_flange, normals_cam)], axis=0)
    nb0 /= np.linalg.norm(nb0)
    x0 = np.concatenate([Rotation.from_matrix(R_fc_init).as_rotvec(),
                         [np.arctan2(nb0[1], nb0[0]), np.arccos(np.clip(nb0[2], -1, 1))]])

    def unpack(x):
        th, ph = x[3], x[4]
        return (Rotation.from_rotvec(x[:3]).as_matrix(),
                np.array([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]))

    def residuals(x):
        R_fc, nb = unpack(x)
        return np.concatenate([Rb @ R_fc @ n - nb for Rb, n in zip(R_base_flange, normals_cam)])

    sol = least_squares(residuals, x0)
    R_fc, nb = unpack(sol.x)
    per_pose = [float(np.degrees(np.arccos(np.clip((Rb @ R_fc @ n) @ nb, -1, 1))))
                for Rb, n in zip(R_base_flange, normals_cam)]
    cond = float(np.linalg.svd(sol.jac[:, :3], compute_uv=False).min())  # observability of R_fc
    return R_fc, nb, per_pose, cond


def solve_translation(T_bf: list, R_fc, nb, offsets: list, landmarks: list):
    """Linear least squares for camera position t_fc, object point p and board height h.

    landmarks[i] may be None (object not seen in that pose): then only its plane row is used.
    Returns t_fc, p, h, residual RMS (mm), std of t_fc (mm).
    """
    rows, rhs = [], []
    for T, c, x in zip(T_bf, offsets, landmarks):
        R, tb = T[:3, :3], T[:3, 3]
        if x is not None:
            A = np.zeros((3, 7))
            A[:, :3], A[:, 3:6] = R, -np.eye(3)
            rows.append(A)
            rhs.append(-(R @ R_fc @ x + tb))
        a = np.zeros((1, 7))
        a[0, :3], a[0, 6] = R.T @ nb, -1.0
        rows.append(a)
        rhs.append(np.array([-(c + nb @ tb)]))
    A, b = np.vstack(rows), np.concatenate(rhs)
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = A @ u - b
    dof = max(1, len(b) - 7)
    sigma2 = float(res @ res) / dof
    std_t = np.sqrt(np.diag(sigma2 * np.linalg.pinv(A.T @ A))[:3])
    return u[:3], u[3:6], float(u[6]), float(np.sqrt(np.mean(res ** 2))), std_t


def predict_landmark(T, R_fc, t_fc, x):
    return T[:3, :3] @ (R_fc @ x + t_fc) + T[:3, 3]


def leave_one_out(T_bf, normals, offsets, landmarks, R_init):
    """Per pose with the object in view: refit without it, then predict that pose's object point."""
    errors = []
    idx = [i for i, x in enumerate(landmarks) if x is not None]
    for j in idx:
        keep = [i for i in range(len(T_bf)) if i != j]
        R_fc, nb, _, _ = solve_rotation([T_bf[i][:3, :3] for i in keep], [normals[i] for i in keep], R_init)
        t, p, _, _, _ = solve_translation([T_bf[i] for i in keep], R_fc, nb,
                                          [offsets[i] for i in keep], [landmarks[i] for i in keep])
        errors.append(float(np.linalg.norm(predict_landmark(T_bf[j], R_fc, t, landmarks[j]) - p)))
    return errors


# ── robot routine ───────────────────────────────────────────────────────
def calibration_poses(T_flange0: np.ndarray, centre: np.ndarray, tilt_deg: float, directions: int = 8):
    """The start pose plus poses tilted `tilt_deg` about horizontal axes through `centre`
    (the object), in `directions` evenly spaced directions: the object stays in view."""
    poses = [T_flange0]
    for k in range(directions):
        a = 2 * np.pi * k / directions
        R = Rotation.from_rotvec(np.radians(tilt_deg) * np.array([np.cos(a), np.sin(a), 0.0])).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R @ T_flange0[:3, :3]
        T[:3, 3] = R @ (T_flange0[:3, 3] - centre) + centre
        poses.append(T)
    return poses


def save(R_fc, t_fc, nb, summary, path=CALIB_FILE):
    from cube_pick.robot.pick_cube import T_to_pose
    T = np.eye(4)
    T[:3, :3] = R_fc
    data = {
        "cam_rpy": [round(v, 4) for v in T_to_pose(T)[3:]],
        "cam_xyz": None if t_fc is None else [round(float(v), 3) for v in t_fc],
        "board_normal_base": np.round(nb, 5).tolist(),
        "board_tilt_deg": round(float(np.degrees(np.arccos(abs(nb[2])))), 2),
        **summary,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def load_cam_calib(default_rpy, default_xyz):
    """(rpy, xyz, description) of the camera tool: calibrated values where available."""
    try:
        data = json.loads(CALIB_FILE.read_text())
        rpy = [float(v) for v in data["cam_rpy"]]
        xyz = [float(v) for v in data["cam_xyz"]] if data.get("cam_xyz") else list(default_xyz)
        what = "rotation + position" if data.get("cam_xyz") else "rotation only (position from TOOL_COORDS)"
        return rpy, xyz, f"{CALIB_FILE.name} ({what}, {data.get('date', '?')})"
    except (OSError, KeyError, ValueError, TypeError):
        return list(default_rpy), list(default_xyz), "CAM_RPY_DEFAULT + TOOL_COORDS (no calibration file)"


def load_cam_rpy(default):  # backwards compatible
    return load_cam_calib(default, [0, 0, 0])[0], load_cam_calib(default, [0, 0, 0])[2]


def run(args):
    """Visit tilted poses around the object, measure board + object, solve, check, save."""
    from cube_pick.config import CAM_TOOL, TOOL_COORDS, clear_faults, connect_robot, ensure_automatic
    from cube_pick.robot.pick_cube import CAM_RPY_DEFAULT, Cell, T_to_pose, pose_to_T
    from suction_pick.vision.camera import capture
    from suction_pick.vision.geometry import depth_to_points

    robot = connect_robot()
    time.sleep(0.5)
    try:
        clear_faults(robot)
    except RuntimeError as exc:
        raise SystemExit(f"[ABORT] {exc}")
    rpy0, xyz0, source = load_cam_calib(CAM_RPY_DEFAULT, TOOL_COORDS[CAM_TOOL][:3])
    print(f"[CALIB] starting from {source}: rpy {np.round(rpy0, 2).tolist()}, xyz {np.round(xyz0, 2).tolist()}")
    cell = Cell(robot, CAM_TOOL, args.suction_tool, 0.0, rpy0, cam_xyz=xyz0)

    # Where is the object now? Tilt around it so it stays in view.
    cell.wait_still()
    T0 = cell.flange()
    _, depth, K = capture(args.frames)
    pc = depth_to_points(depth, K)
    n0, c0, frac0, _ = fit_board_plane(pc, args.max_range)
    x0, info0 = find_landmark(pc, n0, c0, args.max_range)
    if x0 is None:
        print("[CALIB] No object fully in view: calibrating ROTATION only (put the foam disc in view "
              "for the camera position too).")
        centre = (T0 @ cell.T_flange_optical)[:3, 3] + abs(c0) * (T0 @ cell.T_flange_optical)[:3, 2]
    else:
        print(f"[CALIB] object found: {info0['area_px']} px, top {info0['top_mm']} mm above the board, "
              f"image centre {info0['pixel']}")
        centre = (T0 @ cell.T_flange_optical @ np.append(x0, 1.0))[:3]
    poses = calibration_poses(T0, centre, args.tilt)
    targets = [T_to_pose(T @ cell.T_flange_gripper) for T in poses]   # as suction-tool poses
    for i, t in enumerate(targets):
        if cell.reachable(t) is None:
            raise SystemExit(f"[ABORT] calibration pose {i} is unreachable; try a smaller --tilt.")
    print(f"[CALIB] {len(poses)} poses: this one, then tilted {args.tilt:g} deg around the "
          f"{'object' if x0 is not None else 'board'} in {len(poses) - 1} directions "
          f"(joint moves, {args.vel:g}% speed). Board and object must stay still.")
    if not args.yes and input("Type 'y' to start: ").strip().lower() != "y":
        print("Cancelled.")
        return 0
    try:
        ensure_automatic(robot)
    except RuntimeError as exc:
        raise SystemExit(f"[ABORT] {exc}")

    T_bf, normals, offsets, landmarks = [], [], [], []
    try:
        for i, t in enumerate(targets):
            if i:
                cell.move_ptp(t, args.vel, f"calibration pose {i}")
            cell.wait_still()
            T = cell.flange()
            _, depth, K = capture(args.frames)
            pc = depth_to_points(depth, K)
            n, c, frac, rms = fit_board_plane(pc, args.max_range)
            x, info = find_landmark(pc, n, c, args.max_range) if x0 is not None else (None, "-")
            print(f"[CALIB] pose {i}: board {frac:.0%} of view, flat {rms:.1f} mm, camera "
                  f"{np.degrees(np.arccos(-n[2])):.1f} deg off perpendicular | object: "
                  f"{'seen at ' + str(info['pixel']) if x is not None else info}")
            T_bf.append(T)
            normals.append(n)
            offsets.append(c)
            landmarks.append(x)
        cell.move_ptp(targets[0], args.vel, "back to the start pose")
    except KeyboardInterrupt:
        robot.StopMotion()
        print("\n[STOP] Ctrl+C -> StopMotion. Calibration not saved.")
        return 130

    R_init = pose_to_T([0, 0, 0] + list(rpy0))[:3, :3]
    R_fc, nb, per_pose, cond = solve_rotation([T[:3, :3] for T in T_bf], normals, R_init)
    change = np.degrees(Rotation.from_matrix(R_init.T @ R_fc).magnitude())
    print(f"[CALIB] ROTATION: board normals agree within {max(per_pose):.2f} deg (conditioning {cond:.2f}); "
          f"changed by {change:.2f} deg; board tilt in robot base {np.degrees(np.arccos(abs(nb[2]))):.2f} deg")
    ok = max(per_pose) <= 1.0 and cond >= 0.1
    summary = {"rotation_fit_deg_max": round(max(per_pose), 3), "poses": len(T_bf)}
    t_fc = None
    seen = sum(x is not None for x in landmarks)
    if seen >= 4:
        old = [predict_landmark(T, R_fc, np.array(xyz0), x) for T, x in zip(T_bf, landmarks) if x is not None]
        t_fc, p, h, rms_res, std_t = solve_translation(T_bf, R_fc, nb, offsets, landmarks)
        new = [predict_landmark(T, R_fc, t_fc, x) for T, x in zip(T_bf, landmarks) if x is not None]
        scatter_old = float(np.max(np.linalg.norm(np.array(old) - np.mean(old, axis=0), axis=1)))
        scatter_new = float(np.max(np.linalg.norm(np.array(new) - p, axis=1)))
        loo = leave_one_out(T_bf, normals, offsets, landmarks, R_init)
        print(f"[CALIB] POSITION: camera xyz {np.round(t_fc, 1).tolist()} mm (was {np.round(xyz0, 1).tolist()}, "
              f"+/- {np.round(std_t, 1).tolist()})")
        print(f"[CALIB] object seen from {seen} poses lands within {scatter_new:.1f} mm of one point "
              f"(was {scatter_old:.1f} mm with the old camera position)")
        print(f"[CALIB] CHECK (leave-one-out, pose not used in the fit): object predicted within "
              f"mean {np.mean(loo):.1f} mm, max {np.max(loo):.1f} mm")
        summary.update({"object_scatter_mm": round(scatter_new, 2), "object_scatter_before_mm": round(scatter_old, 2),
                        "leave_one_out_mean_mm": round(float(np.mean(loo)), 2),
                        "leave_one_out_max_mm": round(float(np.max(loo)), 2),
                        "cam_xyz_std_mm": np.round(std_t, 2).tolist()})
        ok = ok and np.max(loo) <= 8.0
        if np.max(loo) > 3.0:
            print("[CALIB] Note: leave-one-out above 3 mm; use a larger --tilt or keep the object "
                  "near the image centre for a tighter result.")
    else:
        print(f"[CALIB] Object seen in only {seen} poses (need 4): camera POSITION not calibrated.")
    if not ok:
        print("[CALIB] Poor fit (board/object moved, too little tilt, or not flat): NOT saved.")
        return 1
    data = save(R_fc, t_fc, nb, summary)
    print(f"[CALIB] Saved {CALIB_FILE.name}: cam_rpy {data['cam_rpy']}, cam_xyz {data['cam_xyz']}")
    return 0
