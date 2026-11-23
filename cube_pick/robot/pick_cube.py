#!/usr/bin/env python3
"""Pick the cube using the 6D pose from FoundationPose (cube_pick.vision.live_tracker).

The tracker must be running (start_pick.py starts it); it keeps writing
debug/live/latest_pose.json (camera frame, metres).

Chain used here (Fairino poses are [x, y, z, rx, ry, rz] in mm / degrees,
rotation R = Rz(rz) @ Ry(ry) @ Rx(rx)):

    T_base_camera = T_base_flange @ T_flange_camTCP @ R_camTCP_optical
    T_base_cube   = T_base_camera @ T_camera_obb
    MoveL(gripper-TCP target, tool=<gripper TCP id>)

Usage
  1. Check only (no motion, prints the cube in the robot base frame):
       python -m cube_pick.robot.pick_cube --cam_tool 5 --gripper_tool 1 --check
  2. Dry run (moves nothing, prints hover/grasp targets):
       python -m cube_pick.robot.pick_cube --cam_tool 5 --gripper_tool 1
  3. Pick:
       python -m cube_pick.robot.pick_cube --cam_tool 5 --gripper_tool 1 --execute   (add --hover_only to stop above the cube)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from cube_pick.config import POSE_JSON, active_tool, connect_robot, ensure_automatic, tool_coord

# Measured camera optical-frame rotation relative to the flange (Rx, Ry, Rz deg), from the
# table plane (depth) + cube_calibration.json. Replaces the camera tool's own Rx/Ry/Rz;
# its X/Y/Z lens position still comes from the controller. Use --use_tool_rotation to
# trust the pendant's Rx/Ry/Rz instead.
CAM_RPY_DEFAULT = (22.96, 1.09, 45.3)

# ── grasp / motion tuning ────────────────────────────────────────────────
APPROACH_MM = 80.0          # hover this far above the grasp point before descending
LIFT_MM = 100.0             # lift after closing
MIN_HOVER_Z_MM = 250.0      # hover and lift never go below this base-frame Z
GRASP_Z_OFFSET_MM = 0.0     # gripper TCP z relative to the grasp height (+ = higher)
GRASP_HEIGHT_FRACTION = 0.5 # grip at this fraction of the object's measured height
TABLE_CLEARANCE_MM = 3.0    # gripper TCP must stay this far above the table
# Table and object heights are measured by the camera (support plane + object top from
# depth, written by the tracker); nothing is hard-coded. Sanity limits:
MAX_TABLE_SLOPE_DEG = 15.0  # detected support surface must be roughly horizontal
HEIGHT_MISMATCH_MM = 10.0   # warn when measured height differs from the mesh by more
NOT_RESTING_MM = 15.0       # measured height > mesh + this -> object is not on that surface
GRIPPER_DOWN_RX, GRIPPER_DOWN_RY = 180.0, 0.0   # gripper pointing straight down
GRIPPER_YAW_OFFSET_DEG = 0.0  # extra yaw if the fingers close along the tool Y axis (try 90)
MAX_TILT_DEG = 15.0         # with --check_tilt: refuse cubes that are not lying flat
ASSUME_FLAT_ON_TABLE = True # cube rests on the table -> treat as flat, use only its yaw
# Reach is decided by the controller's inverse kinematics for hover, grasp and lift
# (Cell.reachable), not by a fixed radius, so table corners the arm can reach are allowed.
VEL_TRAVEL, VEL_DESCEND, VEL_LIFT = 20.0, 8.0, 15.0

# ── pose sampling ────────────────────────────────────────────────────────
SETTLE_S = 0.6              # wait after the arm stops before trusting frames
SAMPLES = 10
MIN_DEPTH_FIT = 0.75
MAX_SPREAD_MM = 3.0
SAMPLE_TIMEOUT_S = 8.0


def pose_to_T(pose):
    """Fairino [x,y,z,rx,ry,rz] (mm, deg) -> 4x4 (mm)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T):
    return [*T[:3, 3].tolist(), *Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True).tolist()]


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def call(name, result):
    """Fairino calls return err or (err, value); raise on a non-zero error code."""
    err, value = (result, None) if isinstance(result, int) else (result[0], result[1] if len(result) > 1 else None)
    if err != 0:
        raise RuntimeError(f"{name} failed with error code {err}")
    return value


class Cell:
    def __init__(self, robot, cam_tool, gripper_tool, cam_yaw_deg, cam_rpy=None, cam_xyz=None):
        self.robot = robot
        self.gripper_tool = gripper_tool
        cam_coord = tool_coord(cam_tool)  # from TOOL_COORDS (controller has no GetToolCoordWithID)
        if cam_xyz is not None:  # calibrated lens position replaces the taught one
            cam_coord[:3] = [float(v) for v in cam_xyz]
        if cam_rpy is not None:
            # Keep the taught lens position, replace the orientation with the measured
            # optical frame (Z out of the lens, X image-right, Y image-down).
            print(f"[CAM] tool {cam_tool} position {np.round(cam_coord[:3], 2).tolist()}, "
                  f"orientation {np.round(cam_coord[3:], 2).tolist()} -> override {list(cam_rpy)}")
            cam_coord[3:] = cam_rpy
        T_flange_cam = pose_to_T(cam_coord)
        # Fairino TCP calibration often keeps flange orientation; rotate into the
        # optical frame (X right, Y down, Z forward) when the camera is mounted rotated.
        R_fix = np.eye(4)
        R_fix[:3, :3] = Rotation.from_euler("z", cam_yaw_deg, degrees=True).as_matrix()
        self.T_flange_optical = T_flange_cam @ R_fix
        self.T_flange_gripper = pose_to_T(tool_coord(gripper_tool))  # suction cup TCP
        self._check_convention(cam_tool)

    def _check_convention(self, cam_tool):
        """Cross-check TOOL_COORDS and our Euler maths against the controller (active tool)."""
        active, offset = active_tool(self.robot)
        if active in (cam_tool, self.gripper_tool):
            stored = tool_coord(active)
            dp = np.linalg.norm(np.subtract(stored[:3], offset[:3]))
            dr = np.degrees(Rotation.from_matrix(pose_to_T(stored)[:3, :3].T @ pose_to_T(offset)[:3, :3]).magnitude())
            if dp > 0.5 or dr > 0.5:
                raise RuntimeError(
                    f"Tool {active} on the controller is {np.round(offset, 3).tolist()} but TOOL_COORDS in "
                    f"cube_pick/config.py says {stored}. Update TOOL_COORDS[{active}] first.")
        if active == 0:
            print("[INFO] Active tool 0 (flange); Euler-convention self-check skipped.")
            return
        tool_T = pose_to_T(offset)
        ours = self.flange() @ tool_T
        theirs = pose_to_T(call("GetActualTCPPose", self.robot.GetActualTCPPose()))
        dp = np.linalg.norm(ours[:3, 3] - theirs[:3, 3])
        dr = np.degrees(Rotation.from_matrix(ours[:3, :3].T @ theirs[:3, :3]).magnitude())
        print(f"[CHECK] flange·tool vs controller TCP: {dp:.2f} mm, {dr:.2f}°")
        if dp > 1.0 or dr > 0.5:
            raise RuntimeError("Euler convention / tool data mismatch; do not move the robot.")

    def flange(self):
        return pose_to_T(call("GetActualToolFlangePose", self.robot.GetActualToolFlangePose()))

    def gripper_pose(self):
        return T_to_pose(self.flange() @ self.T_flange_gripper)

    def reachable(self, gripper_pose, ref=None):
        """Joint solution for a gripper-TCP pose, or None.

        The controller's IK solves for whichever tool is active on the pendant (e.g. the
        camera tool 5), so the target is re-expressed for that tool first: no need to
        switch the pendant back to tool 0.
        """
        T_flange = pose_to_T(gripper_pose) @ np.linalg.inv(self.T_flange_gripper)
        T_flange_active = pose_to_T(active_tool(self.robot)[1])  # live: what the IK uses
        target = T_to_pose(T_flange @ T_flange_active)
        # ref = joint state to stay close to (same configuration); None = the current joints
        result = (self.robot.GetInverseKin(0, target, -1) if ref is None
                  else self.robot.GetInverseKinRef(0, target, [float(v) for v in ref]))
        err, joints = (result, None) if isinstance(result, int) else result[:2]
        return joints if err == 0 and joints is not None else None

    def wait_still(self):
        deadline = time.monotonic() + 30
        while call("GetRobotMotionDone", self.robot.GetRobotMotionDone()) != 1:
            if time.monotonic() > deadline:
                raise TimeoutError("Robot did not stop")
            time.sleep(0.05)
        time.sleep(SETTLE_S)

    def move(self, pose, vel, label):
        # Pass the joint solution explicitly. Without it the SDK's MoveL computes IK for the
        # tool ACTIVE on the pendant (e.g. tool 0 = flange), not for `tool=gripper_tool`; the
        # joints then put the flange (not the gripper TCP) at `pose`, and the controller
        # rejects the target with error 74 ("line instruction point error").
        joints = self.reachable(pose)
        if joints is None:
            raise RuntimeError(f"MoveL {label}: no IK solution for {np.round(pose, 1).tolist()}")
        print(f"[MOVE] {label}: {np.round(pose, 1).tolist()} (tool {self.gripper_tool}, vel {vel})")
        call(f"MoveL {label}", self.robot.MoveL(pose, tool=self.gripper_tool, user=0,
                                                joint_pos=list(joints), vel=vel))
        self.wait_still()

    def move_ptp(self, pose, vel, label):
        """Joint-space (PTP) move to a gripper pose. Use for long free-space travel: unlike MoveL
        it may change the arm's joint configuration on the way (MoveL then faults 1/33)."""
        joints = self.reachable(pose)
        if joints is None:
            raise RuntimeError(f"MoveJ {label}: no IK solution for {np.round(pose, 1).tolist()}")
        print(f"[MOVE] {label} (PTP): {np.round(pose, 1).tolist()} (tool {self.gripper_tool}, vel {vel})")
        call(f"MoveJ {label}", self.robot.MoveJ(list(joints), self.gripper_tool, 0, desc_pos=list(pose), vel=vel))
        self.wait_still()

    def gripper(self, closed):
        call("ActGripper", self.robot.ActGripper(2, 1))
        call("MoveGripper", self.robot.MoveGripper(2, 100 if closed else 0, 100, 67, 2000, 0, 0, 0, 0, 0))
        time.sleep(1.0)


def read_cube_samples(after_ns, n=SAMPLES, path=POSE_JSON):
    """Collect n distinct valid FoundationPose results captured after `after_ns`."""
    samples, seen = [], set()
    deadline = time.monotonic() + SAMPLE_TIMEOUT_S
    while len(samples) < n:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Only {len(samples)}/{n} fresh valid poses from {path}. "
                               "Is the tracker running and the cube in view?")
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        stamp = record.get("frame_received_unix_ns")
        if (record.get("valid") and stamp and stamp > after_ns and stamp not in seen
                and record["quality"].get("depth_fit_fraction", 0) >= MIN_DEPTH_FIT):
            seen.add(stamp)
            T = np.array(record["T_camera_obb"])
            T[:3, 3] *= 1000.0  # metres -> mm
            plane = record.get("support_plane_camera")
            if plane is not None:
                plane = (np.array(plane["normal"]), plane["offset_m"] * 1000.0)
            top = record.get("object_top_height_m")
            samples.append((T, np.array(record["extents_m"]) * 1000.0, plane,
                            None if top is None else top * 1000.0))
        time.sleep(0.02)
    return samples


def table_in_base(cell, samples, xy):
    """Table z under base point xy and object top height, both measured from depth.

    Uses the support plane the tracker fits around the object (camera frame,
    n·p + d = 0, n toward the camera). Returns (table_z, top_height_mm, slope_deg) as
    medians over the samples, or None when no plane was measured.
    """
    T = cell.flange() @ cell.T_flange_optical
    zs, tops, slopes = [], [], []
    for _, _, plane, top in samples:
        if plane is None or top is None:
            continue
        normal, offset = plane
        n_b = T[:3, :3] @ normal
        p0 = T[:3, :3] @ (-offset * normal) + T[:3, 3]   # plane point closest to camera
        if abs(n_b[2]) < 1e-3:
            continue
        zs.append(p0[2] - (n_b[0] * (xy[0] - p0[0]) + n_b[1] * (xy[1] - p0[1])) / n_b[2])
        tops.append(top)
        slopes.append(math.degrees(math.acos(min(1.0, abs(n_b[2])))))
    if not zs:
        return None
    return float(np.median(zs)), float(np.median(tops)), float(np.median(slopes))


def cube_in_base(cell, samples):
    T_base_optical = cell.flange() @ cell.T_flange_optical
    cubes = [T_base_optical @ s[0] for s in samples]
    centres = np.array([T[:3, 3] for T in cubes])
    centre = centres.mean(axis=0)
    spread = np.linalg.norm(centres - centre, axis=1).max()
    if spread > MAX_SPREAD_MM:
        raise RuntimeError(f"Pose samples disagree by {spread:.1f} mm; cube or camera moving?")

    # Cube axis closest to vertical -> tilt. Yaw (mod 90 for a cube) comes from BOTH side
    # axes projected onto the table plane, so a spurious pose tilt barely affects it.
    yaws4, tilts = [], []
    for T in cubes:
        R = T[:3, :3]
        up = int(np.argmax(np.abs(R[2])))
        tilts.append(math.degrees(math.acos(min(1.0, abs(R[2, up])))))
        for k, offset in ((1, 0.0), (2, -math.pi / 2)):
            side = R[:, (up + k) % 3]
            yaws4.append(4 * (math.atan2(side[1], side[0]) + offset))
    yaw = math.degrees(math.atan2(np.mean(np.sin(yaws4)), np.mean(np.cos(yaws4))) / 4)
    up = int(np.argmax(np.abs(cubes[0][:3, :3][2])))
    height = float(samples[0][1][up])
    return centre, yaw, float(np.mean(tilts)), height, spread


def grasp_height(centre, model_height, table, manual_table_z=None, source="pose"):
    """Choose the grasp z and the safety floor; works at any object height.

    pose (default): FoundationPose's object-centre z (all depth pixels + the mesh), valid
              wherever the object is: table, stand, shelf or hand.
    measured: detected support surface + GRASP_HEIGHT_FRACTION x measured top height; only
              when the object rests on that surface (thin-walled parts read low).
    Floor: the object's own underside from the pose (always known), raised to the detected
    support surface / --table_z when those lie under the object. Returns (grasp_z, floor_z).
    """
    pose_z = centre[2]
    bottom = pose_z - model_height / 2
    floor_z, floor_name = bottom, "object underside"
    surface_z, top = manual_table_z, None
    if table is not None:
        table_z, top, slope = table
        if slope > MAX_TABLE_SLOPE_DEG:
            raise RuntimeError(f"Detected support surface is tilted {slope:.1f}°; not a table.")
        surface_z = manual_table_z if manual_table_z is not None else table_z
        if manual_table_z is not None:
            print(f"[HEIGHT] --table_z {manual_table_z:.1f} overrides the detected "
                  f"{table_z:.1f} ({table_z - manual_table_z:+.1f} mm)")
    if surface_z is not None:
        gap = bottom - surface_z
        if gap < -NOT_RESTING_MM:
            # A "surface" above the object's underside is not what it stands on.
            print(f"[WARN] Detected surface z {surface_z:.1f} is above the object's underside "
                  f"{bottom:.1f}; ignored.")
            surface_z, top = None, None
        elif gap > NOT_RESTING_MM:
            print(f"[HEIGHT] Object is raised {gap:.0f} mm above the detected surface "
                  f"(z {surface_z:.1f}): on a stand, shelf or in a hand.")
            top = None  # measured top is relative to the wrong surface
        else:
            print(f"[HEIGHT] Object rests on the detected surface z {surface_z:.1f} "
                  f"(underside {bottom:.1f}, {gap:+.1f} mm).")
            if top is not None and abs(top - model_height) > HEIGHT_MISMATCH_MM:
                print(f"[INFO] Depth-measured top {top:.1f} vs mesh {model_height:.1f} mm "
                      "(thin rims read low; the pose height is used unless --height_source measured).")
        if surface_z is not None and surface_z > floor_z:
            floor_z, floor_name = surface_z, "support surface"
    else:
        print("[HEIGHT] No support surface measured; the object's underside is the floor.")
    if source == "measured" and surface_z is not None and top is not None:
        grasp_z = surface_z + GRASP_HEIGHT_FRACTION * top
    else:
        source, grasp_z = "pose", pose_z
    grasp_z += GRASP_Z_OFFSET_MM
    print(f"[HEIGHT] grasp z {grasp_z:.1f} from {source} (object centre z {pose_z:.1f}, "
          f"underside {bottom:.1f}) | floor {floor_z:.1f} = {floor_name}")
    return grasp_z, floor_z


def plan(cell, centre, cube_yaw, grasp_z, floor_z):
    current_rz = cell.gripper_pose()[5]
    # Four symmetric choices; take the one needing the smallest wrist turn.
    rz = min((wrap180(cube_yaw + GRIPPER_YAW_OFFSET_DEG + k * 90) for k in range(4)),
             key=lambda a: abs(wrap180(a - current_rz)))
    grasp = [centre[0], centre[1], grasp_z, GRIPPER_DOWN_RX, GRIPPER_DOWN_RY, rz]
    hover = grasp[:2] + [max(grasp[2] + APPROACH_MM, MIN_HOVER_Z_MM)] + grasp[3:]
    lift = grasp[:2] + [max(grasp[2] + LIFT_MM, MIN_HOVER_Z_MM)] + grasp[3:]

    if floor_z is not None and grasp[2] < floor_z + TABLE_CLEARANCE_MM:
        raise RuntimeError(f"Grasp z {grasp[2]:.1f} is below the safety floor "
                           f"{floor_z:.1f} + {TABLE_CLEARANCE_MM} mm; height looks wrong.")
    return hover, grasp, lift


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam_tool", type=int, required=True, help="Fairino tool id of the camera-centre TCP")
    p.add_argument("--gripper_tool", type=int, required=True, help="Fairino tool id of the gripper TCP")
    p.add_argument("--cam_yaw", type=float, default=0.0, choices=(0.0, 90.0, 180.0, -90.0),
                   help="Rotation about the camera TCP Z to reach the optical frame (see --check)")
    p.add_argument("--cam_rpy", type=float, nargs=3, metavar=("RX", "RY", "RZ"),
                   default=CAM_RPY_DEFAULT,
                   help=f"Camera optical-frame orientation (deg) relative to the flange "
                        f"(default: measured {CAM_RPY_DEFAULT})")
    p.add_argument("--use_tool_rotation", action="store_true",
                   help="Use the camera tool's Rx/Ry/Rz from the pendant instead of --cam_rpy")
    table = p.add_mutually_exclusive_group()
    table.add_argument("--table_z", type=float,
                       help="Optional: override the camera-detected table with a Z (mm) read "
                            "with the GRIPPER tool active and its tip on the table")
    table.add_argument("--table_z_flange", type=float,
                       help="Optional: same, read with tool 0 (flange) while the gripper tip "
                            "touches the table")
    p.add_argument("--height_source", choices=("pose", "measured"), default="pose",
                   help="pose (default): FoundationPose object-centre z, any height; "
                        "measured: detected surface + depth-measured object height")
    p.add_argument("--check", action="store_true", help="No motion: keep printing the cube in base frame")
    p.add_argument("--execute", action="store_true", help="Actually move the robot")
    p.add_argument("--ignore_tilt", action="store_true", help="With --check_tilt: warn instead of refusing")
    p.add_argument("--check_tilt", action="store_true",
                   help="Do not assume the cube is flat on the table; refuse tilts > MAX_TILT_DEG")
    p.add_argument("--hover_only", action="store_true", help="With --execute: go to the hover point and stop")
    p.add_argument("--go_home",action="store_true", help="Move to home_pos first to look at the table")
    return p.parse_args()


def main():
    args = parse_args()
    robot = connect_robot()
    cell = Cell(robot, args.cam_tool, args.gripper_tool, args.cam_yaw,
                None if args.use_tool_rotation else args.cam_rpy)

    if args.check:
        print("[CHECK] Ctrl+C to stop. Turn J6 by ~90° by hand/pendant: a correct setup keeps "
              "the base XYZ fixed (within a few mm).")
        try:
            while True:
                samples = read_cube_samples(time.time_ns(), n=3)
                centre, yaw, tilt, _, spread = cube_in_base(cell, samples)
                table = table_in_base(cell, samples, centre[:2])
                heights = ("no table plane" if table is None else
                           f"table z {table[0]:.1f} | top {table[1]:.1f} mm above it")
                print(f"cube base XYZ {np.round(centre, 1)} mm | yaw {yaw:+.1f}° | tilt {tilt:.1f}° "
                      f"| spread {spread:.1f} mm | {heights}")
        except KeyboardInterrupt:
            return 0

    if args.table_z_flange is not None:
        # Gripper pointing down (rx=180, ry=0): tip z = flange z - tool offset along flange Z.
        tip = pose_to_T([0, 0, 0, GRIPPER_DOWN_RX, GRIPPER_DOWN_RY, 0]) @ cell.T_flange_gripper
        args.table_z = args.table_z_flange + float(tip[2, 3])
        print(f"[TABLE] flange z {args.table_z_flange:.1f} -> gripper-tip table z {args.table_z:.1f} "
              f"(gripper tool length {-tip[2, 3]:.1f} mm)")
    if args.execute:
        ensure_automatic(robot)  # SDK MoveL only runs in automatic mode; manual is for jogging
    else:
        print(f"[ROBOT] Mode: {'automatic' if robot.robot_state_pkg.robot_mode == 0 else 'manual'}")
    if args.go_home and args.execute:
        from cube_pick.robot.home_pos import home_pos
        home_pos(robot)
    if args.execute or args.go_home:
        cell.wait_still()
    else:
        time.sleep(SETTLE_S)

    flange_before = cell.flange()
    samples = read_cube_samples(time.time_ns())
    centre, yaw, tilt, height, spread = cube_in_base(cell, samples)
    table = table_in_base(cell, samples, centre[:2])
    if np.linalg.norm(cell.flange()[:3, 3] - flange_before[:3, 3]) > 0.5:
        raise RuntimeError("Robot moved while sampling the cube pose.")
    print(f"[CUBE] base XYZ {np.round(centre, 1)} mm, yaw {yaw:+.1f}°, tilt {tilt:.1f}°, "
          f"height {height:.1f} mm, spread {spread:.1f} mm")
    if ASSUME_FLAT_ON_TABLE and not args.check_tilt:
        # A cube resting on the table cannot be tilted; FoundationPose tilt here is pose
        # error (e.g. mesh size mismatch). Snap to flat: only yaw is used, gripper points down.
        if tilt > 5.0:
            print(f"[INFO] FoundationPose reports {tilt:.1f}° tilt; cube assumed flat on the "
                  "table (check the mesh dimensions if this stays large).")
    elif tilt > MAX_TILT_DEG:
        if not args.ignore_tilt:
            raise RuntimeError(f"Cube tilted {tilt:.1f}° (> {MAX_TILT_DEG}°); top-down grasp refused.")
        print(f"[WARN] Ignoring tilt {tilt:.1f}° (--ignore_tilt); gripper stays pointing straight down.")

    grasp_z, floor_z = grasp_height(centre, height, table, args.table_z, args.height_source)
    hover, grasp, lift = plan(cell, centre, yaw, grasp_z, floor_z)
    unreachable = []
    for name, pose in (("hover", hover), ("grasp", grasp), ("lift", lift)):
        joints = cell.reachable(pose)
        reach = (f"reachable, joints {np.round(joints, 1).tolist()}" if joints is not None
                 else "NOT reachable (no IK solution)")
        print(f"[PLAN] {name:5s}: {np.round(pose, 1).tolist()} -> {reach}")
        if joints is None:
            unreachable.append(name)
    distance = math.hypot(centre[0], centre[1])
    print(f"[PLAN] cube is {distance:.0f} mm from the robot base axis")
    if unreachable:
        msg = f"Robot cannot reach the {', '.join(unreachable)} pose(s) (no IK solution)."
        if args.execute:
            sys.exit(f"[ABORT] {msg}")
        print(f"[WARN] {msg}")
    if not args.execute:
        print("[DRY RUN] Nothing moved. Add --execute (optionally --hover_only) to move.")
        return 0

    prompt = "move to HOVER only" if args.hover_only else "execute this pick"
    if input(f"Type 'y' to {prompt}: ").strip().lower() != "y":
        return 0
    try:
        if args.hover_only:
            cell.move(hover, VEL_TRAVEL, "hover")
            print(f"[DONE] At hover, {APPROACH_MM:.0f} mm above the computed cube centre. "
                  "Check by eye whether the gripper is over the cube.")

            return 0
        cell.gripper(closed=False)
        cell.move(hover, VEL_TRAVEL, "hover")
        cell.move(grasp, VEL_DESCEND, "descend")
        cell.gripper(closed=True)
        cell.move(lift, VEL_LIFT, "lift")
    except BaseException:
        robot.StopMotion()
        raise
    print("[DONE] Cube picked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
