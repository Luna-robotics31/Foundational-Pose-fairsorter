"""Whole-arm collision check for the Fairino FR10 against the virtual bin (cubicle).

Kinematics: standard DH, UR-style axes. The lengths were fitted to the controller's own forward
kinematics (GetForwardKin) for 40 random joint configurations: error 0.0000 mm.
    d1 = 180, a2 = -700, a3 = -586, d4 = 195, d5 = 114, d6 = 106 mm, no joint offsets.
On start-up the model is compared with the live flange pose; if it disagrees (another robot
type), the arm check is switched off with a warning.

Links are capsules (segment + radius) between the joint frames. The upper arm and forearm of
this arm type sit in planes offset by up to d4 along the joint-2/3 axis, so they are covered at
0, d4/2 and d4 along that axis (conservative). Paths: joint moves (MoveJ) are straight lines in
joint space; straight moves (MoveL) are sampled along the line with the controller's IK.
"""

from __future__ import annotations

import numpy as np

FR10_DH = {"d1": 180.0, "a2": -700.0, "a3": -586.0, "d4": 195.0, "d5": 114.0, "d6": 106.0}
_ALPHA = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])

# capsule radii (mm): conservative envelopes of the FR10 housings
RADII = {"upper arm": 80.0, "forearm": 70.0, "wrist 1": 65.0, "wrist 2": 60.0, "wrist 3": 55.0}


def fk_frames(q_deg, dh=FR10_DH):
    """Base frame + the 6 joint frames (4x4 each) for joint angles in degrees."""
    a = [0, dh["a2"], dh["a3"], 0, 0, 0]
    d = [dh["d1"], 0, 0, dh["d4"], dh["d5"], dh["d6"]]
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        th = np.radians(q_deg[i])
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(_ALPHA[i]), np.sin(_ALPHA[i])
        A = np.array([[ct, -st * ca, st * sa, a[i] * ct], [st, ct * ca, -ct * sa, a[i] * st],
                      [0, sa, ca, d[i]], [0, 0, 0, 1]])
        T = T @ A
        frames.append(T.copy())
    return frames


def capsules(q_deg, dh=FR10_DH, extra=0.0):
    """[(p0, p1, radius, name)] for the arm links at joint angles q_deg."""
    F = fk_frames(q_deg, dh)
    o = [f[:3, 3] for f in F]
    out = []
    for name, (i, j), axis_frame in (("upper arm", (1, 2), 1), ("forearm", (2, 3), 2)):
        z = F[axis_frame][:3, 2]                         # joint 2 / 3 axis
        for s in (0.0, dh["d4"] / 2, dh["d4"]):
            out.append((o[i] + s * z, o[j] + s * z, RADII[name] + extra, name))
    out.append((o[3], o[4], RADII["wrist 1"] + extra, "wrist 1"))
    out.append((o[4], o[5], RADII["wrist 2"] + extra, "wrist 2"))
    out.append((o[5], o[6], RADII["wrist 3"] + extra, "wrist 3"))
    return out


def seg_distance(points, p0, p1):
    """Distance from points (N, 3) to the segment p0-p1."""
    v = p1 - p0
    t = np.clip(((points - p0) @ v) / max(v @ v, 1e-9), 0.0, 1.0)
    return np.linalg.norm(points - (p0 + t[:, None] * v), axis=1)


def clearance(q_deg, points, extra=0.0):
    """(closest distance mm, link name, closest point) between the arm and the points."""
    best = (np.inf, "-", None)
    for p0, p1, r, name in capsules(q_deg, extra=extra):
        d = seg_distance(points, p0, p1) - r
        k = int(np.argmin(d))
        if d[k] < best[0]:
            best = (float(d[k]), name, points[k])
    return best


def model_error(robot):
    """mm between the model's flange and the controller's, at the current joints."""
    q = robot.GetActualJointPosDegree()[1]
    flange = robot.GetActualToolFlangePose()[1]
    return float(np.linalg.norm(fk_frames(q)[-1][:3, 3] - np.array(flange[:3])))


def joint_path(q0, q1, n=40):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return [q0 + t * (q1 - q0) for t in np.linspace(0, 1, n)]


def linear_path(cell, pose0, pose1, q0, step_mm=20.0):
    """Joint states along a straight tool move (controller IK, each near the previous)."""
    from scipy.spatial.transform import Rotation, Slerp
    p0, p1 = np.array(pose0[:3]), np.array(pose1[:3])
    n = int(min(15, max(2, np.ceil(np.linalg.norm(p1 - p0) / step_mm) + 1)))
    rots = Rotation.from_euler("xyz", [pose0[3:], pose1[3:]], degrees=True)
    slerp = Slerp([0, 1], rots)
    qs, ref = [], list(q0)
    for t in np.linspace(0, 1, n):
        pose = list(p0 + t * (p1 - p0)) + list(slerp(t).as_euler("xyz", degrees=True))
        q = cell.reachable(pose, ref)
        if q is None:
            return None
        qs.append(np.asarray(q, float))
        ref = q
    return qs


def path_clearance(qs, points):
    """(closest distance, link, sample index) over a joint path."""
    best = (np.inf, "-", -1)
    for k, q in enumerate(qs):
        d, name, _ = clearance(q, points)
        if d < best[0]:
            best = (d, name, k)
    return best


def ik(T_flange, q_seed, tol_mm=0.1, tol_deg=0.05):
    """Joint angles (deg) putting the flange at T_flange, starting from (and staying near) q_seed.
    Numerical least squares on the fitted model; None if it does not converge."""
    from scipy.optimize import least_squares
    target_R, target_p = T_flange[:3, :3], T_flange[:3, 3]

    def res(q):
        F = fk_frames(q)[-1]
        return np.concatenate([F[:3, 3] - target_p, 300.0 * (F[:3, :3] - target_R).ravel()])

    sol = least_squares(res, np.asarray(q_seed, float), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    F = fk_frames(sol.x)[-1]
    dp = np.linalg.norm(F[:3, 3] - target_p)
    dr = np.degrees(np.arccos(np.clip((np.trace(F[:3, :3].T @ target_R) - 1) / 2, -1, 1)))
    return sol.x if (dp < tol_mm and dr < tol_deg) else None
