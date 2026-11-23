"""Gripper geometry and planner parameters (all lengths in mm, angles in degrees)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fairino tool numbers and offsets live in cube_pick/config.py (TOOL_COORDS).
import cube_pick.config as _robot_cfg  # noqa: E402

CAM_TOOL = _robot_cfg.CAM_TOOL
SUCTION_TOOL = _robot_cfg.SUCTION_TOOL


@dataclass
class Gripper:
    """Multi-cup vacuum gripper, described in the robot TOOL frame.

    The tool frame (Fairino tool coordinate of the suction gripper) has its origin, the TCP,
    in the plane of the cup lips, and its Z axis pointing out of the gripper towards the
    object. All cups are identical and their centres lie in the tool XY plane (paper, Sec. 4).
    """
    cup_xy: list = field(default_factory=lambda: [[-35.0, 0.0], [35.0, 0.0]])  # 70 mm spacing
    cup_diameter: float = 30.0
    # True only if every cup has its own vacuum valve. With one shared vacuum line a cup that
    # is not on a surface leaks, so every cup must be on a graspable surface.
    independent_cups: bool = False
    body_margin: float = 5.0      # extra clearance around the cups for the collision check
    # End-effector volume ABOVE the cups, for the collision check. Gripper frame: origin = TCP in
    # the cup-lip plane, Z = away from the surface (approach axis), x = tool x, y = -tool y.
    # Each part is a column that extends upward without limit, so it also covers the straight
    # approach / lift and the wrist above the flange: a bin wall inside a column = collision.
    body_xy: list = field(default_factory=lambda: [0.0, 0.0])  # flange axis (set from tool 1)
    body_radius: float = 45.0     # suction tool body + flange + wrist, mm
    body_start: float = 10.0      # the body starts this high above the cup lips, mm
    camera_xyz: list | None = None  # lens centre in the gripper frame (set from calibration)
    camera_radius: float = 50.0   # Gemini 2 (about 90 x 25 x 30 mm) + bracket, mm
    camera_below: float = 30.0    # the camera column starts this far below the lens centre, mm

    @property
    def cup_radius(self) -> float:
        return self.cup_diameter / 2

    @property
    def n_cups(self) -> int:
        return len(self.cup_xy)

    def cups_gripper_frame(self) -> np.ndarray:
        """Cup centres in the planner's gripper frame (Z = surface normal, out of the surface).

        That frame is the tool frame turned 180 deg about X: x' = x, y' = -y, z' = -z.
        """
        xy = np.asarray(self.cup_xy, float)
        return np.column_stack([xy[:, 0], -xy[:, 1], np.zeros(len(xy))])

    def min_active(self) -> int:
        return 2 if self.independent_cups else self.n_cups

    def columns(self) -> list:
        """(cx, cy, radius, z_start) of the end-effector parts above the cups (gripper frame)."""
        cols = [(self.body_xy[0], self.body_xy[1], self.body_radius, self.body_start)]
        if self.camera_xyz is not None:
            cx, cy, cz = self.camera_xyz
            cols.append((cx, cy, self.camera_radius, cz - self.camera_below))
        return cols


def robot_ee_geometry() -> dict:
    """Flange axis and camera lens position in the gripper frame, from the suction tool (tool 1)
    and the camera calibration (or tool 5). Empty dict if the robot config is unavailable."""
    try:
        import cube_pick.config as rc
        from cube_pick.robot.pick_cube import CAM_RPY_DEFAULT, pose_to_T
        from suction_pick.robot.calibrate import load_cam_calib
    except Exception:
        return {}
    T_tool_flange = np.linalg.inv(pose_to_T(rc.TOOL_COORDS[rc.SUCTION_TOOL]))   # flange -> tool, inverted
    _, cam_xyz, _ = load_cam_calib(CAM_RPY_DEFAULT, rc.TOOL_COORDS[rc.CAM_TOOL][:3])
    flip = np.array([1.0, -1.0, -1.0])                                       # tool frame -> gripper frame
    flange = (T_tool_flange @ [0, 0, 0, 1])[:3] * flip
    camera = (T_tool_flange @ [*cam_xyz, 1])[:3] * flip
    return {"body_xy": [float(flange[0]), float(flange[1])], "camera_xyz": [float(v) for v in camera]}


@dataclass
class AffordanceParams:
    """Where a single cup can seal (replaces the paper's learned SG-U-Net++ map)."""
    max_range: float = 1200.0        # ignore depth farther than this from the camera, mm (0 = off)
    flatness: float = 2.0            # max RMS deviation from a plane under the cup, mm ...
    noise_fraction: float = 0.003    # ... or this fraction of the distance, if larger (sensor noise)
    min_valid_fraction: float = 0.95  # depth pixels under the cup that must be valid
    max_tilt: float = 45.0           # surface tilt from "up", deg
    min_height: float = 8.0          # above the table plane; <= 0 disables table removal
    min_region_px: int = 30          # drop smaller graspable regions


@dataclass
class PlannerParams:
    """Multi-cup planner (paper Sec. 5)."""
    voxel: float = 5.0               # l = 0.005 m
    angle_step: float = 5.0          # delta alpha = 5 deg
    normal_tol: float = 11.5         # epsilon_1, Eq. 2
    contact_tol: float = 10.0        # epsilon_2, Eq. 3
    min_points_per_voxel: int = 3    # paper: > 10 (closer camera); fewer points per voxel at 1-1.6 m
    top_normal_fraction: float = 0.10
    roll_range: float = 180.0        # 180 deg covers any centrally symmetric cup layout
    max_tilt: float = 45.0           # gripper approach axis from "up"
    max_objects: int = 1             # objects per grasp: 1 = one parcel at a time (paper: no limit)
    refine_axis: bool = True         # align approach with the measured normal (not in the paper)
    max_seal_gap: float = 4.0        # every active cup within this of its surface, mm
    collision_tol: float = 3.0       # scene points this far above the cup plane = collision
    clearance: float = 30.0          # pre-grasp / lift at least this far above everything near the path
    hover_radius: float = 250.0      # obstacles (bin rim) within this of the approach line set the hover height
    collision_max_points: int = 5
    max_detailed: int = 400          # candidates that get the full (slower) checks
    max_grid_voxels: int = 4_000_000 # safety: refuse absurd scenes instead of crawling
    max_results: int = 10
    nms_distance: float = 10.0       # mm, merge near-duplicate grasps
    orient_weight: float = 10.0      # mm of centre distance worth one unit of J_orient
