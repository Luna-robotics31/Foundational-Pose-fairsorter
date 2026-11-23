"""Geometry of an L-shaped weld seam with a smooth, curvature-continuous corner.

Pure maths, no robot: build the corner, sample the path, report its properties.

Poses are Fairino [x, y, z, rx, ry, rz] in mm / degrees with R = Rz(rz) @ Ry(ry) @ Rx(rx),
i.e. scipy Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).

Corner construction
-------------------
Line 1 is p1 -> p2 and line 2 is p3 -> p4. The taught p2 -> p3 move is the sharp corner
(direction and torch orientation change almost instantly), which is what makes the arm
vibrate. Both lines are extended to the corner where they meet: C1 on line 1 and C2 on
line 2 are the closest points of the two lines (they differ only by a small height gap).
The corner is replaced by a quintic (5th-order) Bezier curve from A = C1 - d*u1 to
B = C2 + d*u2, where d is the corner size ("--corner_mm"):

    Q0 = A,  Q1 = C1 - a*d*u1,  Q2 = C1 - b*d*u1,
    Q3 = C2 + b*d*u2,  Q4 = C2 + a*d*u2,  Q5 = B          (a = 0.6, b = 0.3)

Q0, Q1, Q2 lie on line 1 and Q3, Q4, Q5 on line 2, so at both ends the curve
  * has the same direction as the line (G1: no kink), and
  * has zero curvature (G2: no step in sideways acceleration).
A circular arc (MoveC) or a plain controller blend has a curvature jump at both ends,
which is a jerk spike - the source of vibration this curve avoids.
a, b were chosen to (nearly) minimise the rate of change of curvature along the curve
for a ~90 deg corner (sideways jerk at constant speed is v^3 * dk/ds); putting the inner
points at the corner instead (b = 0) hugs the corner closer but is ~8x harsher.

The torch orientation turns from the line-1 orientation (at A) to the line-2 orientation
(at B) with a smootherstep profile over the curve's arc length, so the angular speed and
angular acceleration are also zero where the curve meets the straight lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MAX_LINE_GAP_MM = 3.0             # the two lines must (almost) meet at the corner
MIN_TURN_DEG, MAX_TURN_DEG = 20.0, 160.0
END_MARGIN_MM = 2.0               # the curve must start/end this far inside each line
MIN_LINE_MM = 5.0
CTRL_A, CTRL_B = 0.6, 0.3         # inner Bezier control points, as fractions of corner_mm


def rot_of(pose) -> Rotation:
    return Rotation.from_euler("xyz", np.asarray(pose[3:6], float), degrees=True)


def pose_of(pos, rot: Rotation) -> list[float]:
    return [float(v) for v in pos] + [float(v) for v in rot.as_euler("xyz", degrees=True)]


def bezier(ctrl: np.ndarray, t) -> np.ndarray:
    """Points of the Bezier curve with control points `ctrl` (n+1, 3) at parameters t."""
    n = len(ctrl) - 1
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    return sum(comb(n, i) * (1 - t) ** (n - i) * t ** i * ctrl[i] for i in range(n + 1))


def bezier_derivative(ctrl: np.ndarray) -> np.ndarray:
    """Control points of the derivative curve."""
    return (len(ctrl) - 1) * (ctrl[1:] - ctrl[:-1])


def smootherstep(s):
    """0 -> 1 with zero 1st and 2nd derivative at both ends."""
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def closest_points(p, u, q, v):
    """Closest points of the lines p + s*u and q + t*v (u, v unit vectors)."""
    w = p - q
    b, d, e = u @ v, u @ w, v @ w
    den = 1.0 - b * b
    if den < 1e-9:
        raise ValueError("The two weld lines are parallel; there is no corner.")
    s = (b * e - d) / den
    t = (e - b * d) / den
    return p + s * u, q + t * v


@dataclass
class Sample:
    pose: list[float]  # Fairino [x, y, z, rx, ry, rz]
    s: float           # path length from p1, mm
    part: str          # "line1", "corner" or "line2"


class Seam:
    """L-shaped seam p1 -> p2 | corner | p3 -> p4 with a smooth Bezier corner."""

    def __init__(self, p1, p2, p3, p4, corner_mm: float):
        poses = [np.asarray(p, float) for p in (p1, p2, p3, p4)]
        for name, p in zip(("p1", "p2", "p3", "p4"), poses):
            if p.shape != (6,) or not np.all(np.isfinite(p)):
                raise ValueError(f"{name} must be 6 finite numbers [x, y, z, rx, ry, rz].")
        self.taught = [p.tolist() for p in poses]
        x = [p[:3] for p in poses]
        self.x = x

        d1, d2 = x[1] - x[0], x[3] - x[2]
        self.len1, self.len2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
        if min(self.len1, self.len2) < MIN_LINE_MM:
            raise ValueError("p1->p2 and p3->p4 must each be at least "
                             f"{MIN_LINE_MM} mm long (got {self.len1:.1f}, {self.len2:.1f}).")
        self.u1, self.u2 = d1 / self.len1, d2 / self.len2
        self.turn_deg = float(np.degrees(np.arccos(np.clip(self.u1 @ self.u2, -1, 1))))
        if not MIN_TURN_DEG <= self.turn_deg <= MAX_TURN_DEG:
            raise ValueError(f"Turn angle {self.turn_deg:.1f} deg is outside "
                             f"{MIN_TURN_DEG:.0f}-{MAX_TURN_DEG:.0f} deg.")

        self.C1, self.C2 = closest_points(x[0], self.u1, x[2], self.u2)
        self.corner = (self.C1 + self.C2) / 2
        self.gap_mm = float(np.linalg.norm(self.C2 - self.C1))
        if self.gap_mm > MAX_LINE_GAP_MM:
            raise ValueError(f"The two lines miss each other by {self.gap_mm:.1f} mm "
                             f"(max {MAX_LINE_GAP_MM} mm). Check the taught points.")

        # How much straight line there is before / after the corner.
        self.room1 = float((self.C1 - x[0]) @ self.u1)
        self.room2 = float((x[3] - self.C2) @ self.u2)
        max_d = min(self.room1, self.room2) - END_MARGIN_MM
        if not 0 < corner_mm <= max_d:
            raise ValueError(f"corner_mm must be in (0, {max_d:.1f}] for these points "
                             f"(got {corner_mm}).")
        self.d = float(corner_mm)
        self.A = self.C1 - self.d * self.u1
        self.B = self.C2 + self.d * self.u2
        a, b = CTRL_A * self.d, CTRL_B * self.d
        self.ctrl = np.array([self.A, self.C1 - a * self.u1, self.C1 - b * self.u1,
                              self.C2 + b * self.u2, self.C2 + a * self.u2, self.B])

        r = [rot_of(p) for p in poses]
        self._slerp1 = Slerp([0, 1], Rotation.concatenate([r[0], r[1]]))
        self._slerp2 = Slerp([0, 1], Rotation.concatenate([r[2], r[3]]))
        self.rot_A = self.rot_line1(self.A)
        self.rot_B = self.rot_line2(self.B)
        self._slerp_corner = Slerp([0, 1], Rotation.concatenate([self.rot_A, self.rot_B]))

        # Dense arc-length table of the corner curve.
        self._t = np.linspace(0, 1, 4001)
        pts = bezier(self.ctrl, self._t)
        self._cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        self.corner_len = float(self._cum[-1])
        self.line1_len = float(np.linalg.norm(self.A - x[0]))
        self.line2_len = float(np.linalg.norm(x[3] - self.B))

    # ── orientation along each part ─────────────────────────────────────
    def rot_line1(self, pos) -> Rotation:
        return self._slerp1(float(np.clip((pos - self.x[0]) @ self.u1 / self.len1, 0, 1)))

    def rot_line2(self, pos) -> Rotation:
        return self._slerp2(float(np.clip((pos - self.x[2]) @ self.u2 / self.len2, 0, 1)))

    def corner_at(self, s_mm) -> tuple[np.ndarray, Rotation]:
        """Position and orientation at arc length s_mm along the corner (0 = A)."""
        s_mm = np.atleast_1d(s_mm)
        t = np.interp(s_mm, self._cum, self._t)
        return bezier(self.ctrl, t), self._slerp_corner(smootherstep(s_mm / self.corner_len))

    # ── sampling ────────────────────────────────────────────────────────
    def corner_samples(self, step_mm: float) -> list[Sample]:
        """A ... B at (almost) equal arc-length spacing <= step_mm, both ends included."""
        n = max(2, ceil(self.corner_len / step_mm))
        s = np.linspace(0, self.corner_len, n + 1)
        pos, rots = self.corner_at(s)
        return [Sample(pose_of(p, r), self.line1_len + si, "corner")
                for p, r, si in zip(pos, rots, s)]

    def line_samples(self, start, end, rot_fn, step_mm, s0, part, include_start):
        length = float(np.linalg.norm(end - start))
        n = max(1, ceil(length / step_mm))
        out = []
        for k in range(0 if include_start else 1, n + 1):
            p = start + (end - start) * k / n
            out.append(Sample(pose_of(p, rot_fn(p)), s0 + length * k / n, part))
        return out

    def dense_path(self, corner_step_mm: float, line_step_mm: float) -> list[Sample]:
        """p1 -> A (line), A -> B (corner), B -> p4 (line); no duplicated points."""
        line1 = self.line_samples(self.x[0], self.A, self.rot_line1, line_step_mm, 0.0,
                                  "line1", include_start=True)
        corner = self.corner_samples(corner_step_mm)[1:]   # A already ends line1
        s_b = self.line1_len + self.corner_len
        line2 = self.line_samples(self.B, self.x[3], self.rot_line2, line_step_mm, s_b,
                                  "line2", include_start=False)
        # Pin the exact taught end orientation at p4 (and p1).
        line1[0].pose = list(self.taught[0])
        line2[-1].pose = list(self.taught[3])
        return line1 + corner + line2

    # ── diagnostics ─────────────────────────────────────────────────────
    def curvature(self, n=2001):
        """(arc length from A, curvature 1/mm) along the corner."""
        t = np.linspace(0, 1, n)
        d1 = bezier(bezier_derivative(self.ctrl), t)
        d2 = bezier(bezier_derivative(bezier_derivative(self.ctrl)), t)
        k = np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1) ** 3
        return np.interp(t, self._t, self._cum), k

    def summary(self, corner_step_mm: float) -> dict:
        _, k = self.curvature()
        pts = bezier(self.ctrl, self._t)
        samples = self.corner_samples(corner_step_mm)
        rates = []
        for a, b in zip(samples[:-1], samples[1:]):
            ang = np.degrees((rot_of(a.pose).inv() * rot_of(b.pose)).magnitude())
            rates.append(ang / max(b.s - a.s, 1e-9))
        return {
            "turn_deg": self.turn_deg,
            "line_gap_mm": self.gap_mm,
            "corner_xyz": self.corner.tolist(),
            "p2_before_corner_mm": float((self.C1 - self.x[1]) @ self.u1),
            "p3_after_corner_mm": float((self.x[2] - self.C2) @ self.u2),
            "torch_turn_deg": float(np.degrees((self.rot_A.inv() * self.rot_B).magnitude())),
            "corner_len_mm": self.corner_len,
            "min_radius_mm": float(1 / k.max()),
            "end_curvature": (float(k[0]), float(k[-1])),
            "closest_to_corner_mm": float(np.linalg.norm(pts - self.corner, axis=1).min()),
            "max_torch_turn_deg_per_mm": float(max(rates)),
            "total_len_mm": self.line1_len + self.corner_len + self.line2_len,
            "corner_samples": len(samples),
        }
