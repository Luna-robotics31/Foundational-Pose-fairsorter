"""Whole-arm collision check for the Fairino FR10 against the virtual bin (cubicle).

Kinematics: standard DH, UR-style axes, equal to the controller's own forward kinematics
(GetForwardKin, checked at random joint configurations: error 0.00 mm). This robot is hardware
V5.0: d1 = 180, a2 = -700, a3 = -586, d4 = 195, d5 = 114, d6 = 106 mm, no joint offsets.
On start-up the model is compared with the live flange pose; if it disagrees (another robot
type), the arm check is switched off with a warning.

Link shapes: the FR10 URDF in robot/fr10/ (same joints as above, Fairino's V5 meshes). Each
link is covered by 1-3 capsules (segment + radius) fitted around its mesh (fr10/capsules.json,
the whole mesh surface is inside them). Paths: joint moves (MoveJ) are straight lines in joint
space; straight moves (MoveL) are sampled along the line with the controller's IK.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

FR10_DH = {"d1": 180.0, "a2": -700.0, "a3": -586.0, "d4": 195.0, "d5": 114.0, "d6": 106.0}
_ALPHA = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
FR10_DIR = Path(__file__).resolve().parent / "fr10"
URDF_FILE = FR10_DIR / "fr10.urdf"


def fk_frames(q_deg, dh=FR10_DH):
    """Base frame + the 6 joint frames (4x4 each, DH convention) for joint angles in degrees."""
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


def _origin(el):
    """4x4 (mm) of a URDF <origin xyz rpy>."""
    T = np.eye(4)
    if el is None:
        return T
    r, p, y = (float(v) for v in el.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = [1000.0 * float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


@lru_cache(maxsize=1)
def urdf_joints():
    """[(name, parent link, child link, T_origin (mm), revolute?)] in chain order."""
    out = []
    for j in ET.parse(URDF_FILE).getroot().findall("joint"):
        axis = j.find("axis")
        if j.get("type") == "revolute" and axis is not None and axis.get("xyz").split() != ["0", "0", "1"]:
            raise ValueError("fr10.urdf: joint axes must be z")
        out.append((j.get("name"), j.find("parent").get("link"), j.find("child").get("link"),
                    _origin(j.find("origin")), j.get("type") == "revolute"))
    return tuple(out)


def joint_transforms(q_deg):
    """[(parent, child, T_parent_child)] for the URDF joints at q_deg (what TF publishes)."""
    out, k = [], 0
    for _, parent, child, T0, revolute in urdf_joints():
        T = T0
        if revolute:
            th = np.radians(q_deg[k])
            k += 1
            R = np.eye(4)
            R[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
            T = T0 @ R
        out.append((parent, child, T))
    return out


def link_frames(q_deg):
    """{link name: 4x4 in the robot base frame (mm)} from the URDF."""
    F = {"base_link": np.eye(4)}
    for parent, child, T in joint_transforms(q_deg):
        F[child] = F[parent] @ T
    return F


@lru_cache(maxsize=1)
def _capsule_table():
    data = json.loads((FR10_DIR / "capsules.json").read_text())["capsules"]
    return tuple((c["link"], np.array(c["p0"], float), np.array(c["p1"], float), float(c["r"]), c["name"])
                 for c in data)


def capsules(q_deg, extra=0.0):
    """[(p0, p1, radius, name)] (robot base frame, mm) covering every moving link at q_deg."""
    F = link_frames(q_deg)
    out = []
    for link, a, b, r, name in _capsule_table():
        T = F[link]
        out.append((T[:3, :3] @ a + T[:3, 3], T[:3, :3] @ b + T[:3, 3], r + extra, name))
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
