"""Live virtual-fence monitor: watch the real robot against the virtual bin walls while it moves.

The planner already rejects grasps whose path would touch a wall. This checks the ACTUAL motion:
during every move a thread reads the live flange pose from the robot's status stream
(robot_state_pkg.flange_cur_pos, ~8 ms updates, no RPC call - safe while the main thread is
blocked in MoveL/MoveJ), places the end-effector model there (cup footprint, tool body + wrist,
camera) and measures the distance to the wall points. It reports the closest approach per move
and sends StopMotion (over its own connection) if any part comes within `margin` mm of a wall.
"""

from __future__ import annotations

import threading
import time
import xmlrpc.client

import numpy as np


class FenceStop(Exception):
    """The robot came too close to a virtual wall and was stopped."""


def part_distances(q: np.ndarray, ee: dict, collision_tol: float):
    """Signed distance (mm) from points q (gripper frame) to each end-effector part.

    Parts extend upward from their start height, like the planner's columns. Negative = inside.
    Returns list of (name, min distance, index of the closest point).
    """
    out = []
    hx, hy = ee["half"]
    dx, dy, dz = np.abs(q[:, 0]) - hx, np.abs(q[:, 1]) - hy, collision_tol - q[:, 2]
    outside = np.linalg.norm(np.maximum(np.column_stack([dx, dy, dz]), 0), axis=1)
    inside = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0)
    d = outside + inside
    out.append(("cups", float(d.min()), int(d.argmin())))
    for (cx, cy, r, z0), name in zip(ee["columns"], ("tool body", "camera")):
        dr = np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r
        dzc = z0 - q[:, 2]
        d = np.where((dr <= 0) & (dzc <= 0), np.maximum(dr, dzc), np.hypot(np.maximum(dr, 0), np.maximum(dzc, 0)))
        out.append((name, float(d.min()), int(d.argmin())))
    return out


class Fence:
    def __init__(self, robot, ip, walls, ee, collision_tol, T_flange_tool, margin=10.0, period=0.02,
                 arm_walls=None):
        self.robot, self.url = robot, f"http://{ip}:20003"
        self.walls, self.ee, self.tol = walls, ee, collision_tol
        self.T_flange_tool, self.margin, self.period = T_flange_tool, margin, period
        self.log = []
        self.arm_walls = arm_walls          # cubicle walls for the whole-arm check (or None)

    def clearance(self, flange_pose):
        """(closest distance mm, part name, wall point) for a flange pose [x y z rx ry rz]."""
        from cube_pick.robot.pick_cube import pose_to_T
        T = pose_to_T(flange_pose) @ self.T_flange_tool          # suction TCP in the base frame
        R = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])               # gripper frame (Z out of the surface)
        q = (self.walls - T[:3, 3]) @ R
        name, d, i = min(part_distances(q, self.ee, self.tol), key=lambda t: t[1])
        return d, name, self.walls[i]

    def _stop(self):
        class _Timeout(xmlrpc.client.Transport):
            def make_connection(self, host):
                conn = super().make_connection(host)
                conn.timeout = 2.0
                return conn
        for _ in range(3):
            try:
                xmlrpc.client.ServerProxy(self.url, transport=_Timeout()).StopMotion()
                return True
            except Exception:
                continue
        return False

    def watch(self, label):
        return _Watch(self, label)


class _Watch:
    def __init__(self, fence, label):
        self.f, self.label = fence, label
        self.best = (np.inf, "-", None)
        self.tripped = None
        self._done = threading.Event()

    def _sample(self):
        st = self.f.robot.robot_state_pkg
        pose = list(st.flange_cur_pos)
        d, part, p = self.f.clearance(pose)                     # end effector vs walls
        if self.f.arm_walls is not None:                        # whole arm vs cubicle, from live joints
            from suction_pick.robot.arm import clearance as arm_clearance
            da, link, pa = arm_clearance(list(st.jt_cur_pos), self.f.arm_walls)
            if da < d:
                d, part, p = da, link, pa
        if d < self.best[0]:
            self.best = (d, part, p)
        if d < self.f.margin and self.tripped is None:
            self.tripped = (d, part, p, pose)
            ok = self.f._stop()
            print(f"\n[FENCE] {self.label}: {part} {d:.0f} mm from the virtual wall at "
                  f"{np.round(p, 0).tolist()} -> StopMotion {'sent' if ok else 'FAILED - PRESS THE E-STOP'}",
                  flush=True)

    def _run(self):
        while not self._done.is_set():
            self._sample()
            time.sleep(self.f.period)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._done.set()
        self._thread.join(timeout=1.0)
        self._sample()                      # always include the pose where the move ended
        d, part, p = self.best
        self.f.log.append({"move": self.label, "closest_mm": round(d, 1), "part": part})
        if self.tripped is None:
            print(f"[FENCE] {self.label}: closest wall {d:.0f} mm ({part})")
        if self.tripped is not None:
            raise FenceStop(f"{self.label}: {self.tripped[1]} came {self.tripped[0]:.0f} mm from the virtual "
                            f"wall (limit {self.f.margin:g} mm); robot stopped")
        return False
