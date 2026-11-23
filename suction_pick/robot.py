"""Suction pick on the Fairino cobot with the wrist Gemini 2 (camera = tool 5, suction = tool 1).

ONE parcel per run. Sequence: go to HOME first (skip: --no_start_home) -> measure there
-> plan in the robot BASE frame -> pre-grasp along the surface normal -> slow approach (small
press for the seal) -> vacuum on -> wait (optionally check a vacuum-sensor input) -> lift
straight up -> optional place + vacuum off -> RETURN TO HOME (HOME_POSE in home_pos.py).
Dry run (no --execute) moves nothing: it plans, checks IK for every pose and prints them.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np

from cube_pick.config import clear_faults, connect_robot, ensure_automatic
from cube_pick.robot.home_pos import HOME_POSE, home_joints, home_pos
from cube_pick.robot.pick_cube import (CAM_RPY_DEFAULT, VEL_DESCEND, VEL_LIFT, VEL_TRAVEL,
                                       Cell, T_to_pose, call, pose_to_T)

from .geometry import depth_to_points, transform
from .fence import Fence, FenceStop


def tool_pose(position, R_tool):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_tool, position
    return T_to_pose(T)


def approach_distance(grasp, args):
    """How far above the parcel (along its normal) the hover point must be: at least
    approach_mm, and `clearance_mm` above the highest thing near the path (e.g. a bin rim), so the
    joint move ends ABOVE the bin and only the straight, collision-checked line goes into it."""
    return max(args.approach_mm, grasp.terms.get("clear_height_mm", 0.0) + args.clearance_mm)


def waypoints(grasp, args):
    """pre-grasp (hover), grasp (pressed), lift poses for one grasp, in the base frame (mm, deg).
    Approach and lift run along the same line (the surface normal), the one checked for collisions."""
    z = grasp.R[:, 2]  # approach normal, out of the surface
    R_tool = grasp.R_tool
    d = approach_distance(grasp, args)
    return {
        "pre-grasp": tool_pose(grasp.tcp + d * z, R_tool),
        "grasp": tool_pose(grasp.tcp - args.press_mm * z, R_tool),
        "lift": tool_pose(grasp.tcp + max(args.lift_mm, d) * z, R_tool),
    }


class Vacuum:
    """Suction output(s) plus an optional release output (e.g. blow-off valve), pulsed on release."""

    def __init__(self, robot, outputs, tool_io, n_cups, release=None, release_time=2.0):
        if len(outputs) not in (1, n_cups):
            raise ValueError(f"--vacuum_do needs 1 output (shared) or {n_cups} (one per cup).")
        self.robot, self.outputs, self.tool_io = robot, outputs, tool_io
        self.release_port, self.release_time = release, release_time

    def _set(self, port, on):
        fn = self.robot.SetToolDO if self.tool_io else self.robot.SetDO
        call(f"{'SetToolDO' if self.tool_io else 'SetDO'}({port}, {int(on)})", fn(port, int(on), 0, 0))

    def release_off(self):
        if self.release_port is not None:
            self._set(self.release_port, False)

    def release(self):
        """Suction OFF and release output ON at the same moment, release OFF after release_time."""
        for port in self.outputs:
            self._set(port, False)
        if self.release_port is None:
            print(f"[VACUUM] OFF (outputs {self.outputs})")
            return
        self._set(self.release_port, True)
        print(f"[VACUUM] suction OFF (DO{self.outputs}) + release ON (DO{self.release_port}) "
              f"for {self.release_time:g} s")
        try:
            time.sleep(self.release_time)
        finally:
            self._set(self.release_port, False)
            print(f"[VACUUM] release OFF (DO{self.release_port})")

    def set(self, on, active=None):
        ports = (self.outputs if len(self.outputs) == 1 or active is None
                 else [o for o, a in zip(self.outputs, active) if a])
        for port in ports:
            fn = self.robot.SetToolDO if self.tool_io else self.robot.SetDO
            call(f"{'SetToolDO' if self.tool_io else 'SetDO'}({port}, {int(on)})", fn(port, int(on), 0, 0))
        print(f"[VACUUM] {'ON' if on else 'OFF'} (outputs {ports}{', tool IO' if self.tool_io else ''})")


def measure_and_plan(cell, args, plan_fn):
    """Capture at the current pose and plan in the base frame."""
    from .camera import capture
    cell.wait_still()
    T_base_cam = cell.flange() @ cell.T_flange_optical  # mm, valid while the arm is still
    bgr, depth, K = capture(args.frames)
    points = transform(T_base_cam, depth_to_points(depth, K))
    return plan_fn(bgr, depth, K, points, T_base_cam)


def equivalent_grasps(g):
    """The grasp, plus the same grasp turned 180 deg about the approach axis if the cup layout
    is symmetric (e.g. 2 identical cups): same cups on the same spots, different wrist angle."""
    from .planner import Grasp
    if not g.terms.get("turned_clear", True):           # turned version would hit something
        return [g]
    local = (g.cups - g.tcp) @ g.R                       # cup offsets in the gripper frame
    perm = []
    for c in local:
        d = np.linalg.norm(local[:, :2] + c[:2], axis=1)  # where -c lies
        if d.min() > 1.0:
            return [g]
        perm.append(int(np.argmin(d)))
    R2 = g.R @ np.diag([-1.0, -1.0, 1.0])
    turned = Grasp(g.tcp, R2, g.tcp + local @ R2.T, g.active[perm], g.cup_labels[perm],
                   g.n_objects, g.score, dict(g.terms))
    return [g, turned]


def linear_chain_problem(cell, poses, names, max_step_deg=30.0, j5_min_deg=5.0):
    """Check that straight-line (MoveL) moves through `poses` stay in ONE joint configuration.

    The first pose is reached by a PTP move, so it may use any configuration; each next pose
    is solved close to the previous solution. A large joint jump or a wrist (J5) near 0 deg
    means the controller would stop the MoveL with fault 1/33/34 (configuration change /
    singular pose). Returns a description of the problem, or None.
    """
    ref = None
    for pose, name in zip(poses, names):
        q = cell.reachable(pose, ref)
        if q is None:
            return f"{name} is unreachable"
        if abs(q[4]) < j5_min_deg:
            return f"{name} is at a wrist singularity (J5 = {q[4]:.1f} deg)"
        if ref is not None:
            step = float(np.max(np.abs(np.subtract(q, ref))))
            if step > max_step_deg:
                return f"straight move to {name} needs a {step:.0f} deg joint change (configuration flip)"
        ref = q
    return None


def make_arm_walls(robot, args):
    """Cubicle walls for the whole-arm check, or None (no bin / --no_bin / --no_arm_check / wrong model)."""
    if args.no_bin or args.no_arm_check:
        return None
    from . import arm, workcell
    bin_ = workcell.load()
    if bin_ is None:
        return None
    err = arm.model_error(robot)
    if err > 1.0:
        print(f"[ARM] WARNING: arm model is {err:.1f} mm off the controller (not an FR10?): whole-arm check OFF")
        return None
    return workcell.arm_wall_points(bin_, args.bin_height)


def plan_route(cell, wps, q_start, q_home, walls, args):
    """Check the whole arm along home -> pre-grasp -> grasp -> lift -> home against the cubicle.
    If too close, retry with a via point straight above the pre-grasp, above the walls.
    Returns (waypoints, report) or (None, reason)."""
    if walls is None:
        return wps, ["whole-arm check off (no bin.json, --no_bin or --no_arm_check)"]
    from . import arm

    def check(w):
        lines, worst = [], np.inf
        q_pre = cell.reachable(w["pre-grasp"])
        if q_pre is None:
            return -np.inf, ["pre-grasp unreachable"]
        moves = []
        if "via" in w:
            q_via = cell.reachable(w["via"])
            if q_via is None:
                return -np.inf, ["via point unreachable"]
            moves.append(("home -> via (joint move)", arm.joint_path(q_start, q_via)))
            moves.append(("via -> pre-grasp", arm.linear_path(cell, w["via"], w["pre-grasp"], q_via)))
        else:
            moves.append(("home -> pre-grasp (joint move)", arm.joint_path(q_start, q_pre)))
        down = arm.linear_path(cell, w["pre-grasp"], w["grasp"], q_pre)
        moves.append(("pre-grasp -> grasp", down))
        up = arm.linear_path(cell, w["grasp"], w["lift"], down[-1]) if down else None
        moves.append(("grasp -> lift", up))
        last = up[-1] if up else None
        if "via" in w and last is not None:
            back = arm.linear_path(cell, w["lift"], w["via"], last)
            moves.append(("lift -> via", back))
            last = back[-1] if back else None
        if last is not None:
            moves.append(("-> home (joint move)", arm.joint_path(last, q_home)))
        for name, qs in moves:
            if qs is None:
                return -np.inf, [f"{name}: no IK solution along the line"]
            d, link, _ = arm.path_clearance(qs, walls)
            worst = min(worst, d)
            lines.append(f"{name}: {d:.0f} mm ({link})")
        return worst, lines

    d, lines = check(wps)
    if d >= args.arm_margin:
        return wps, lines
    via = list(wps["pre-grasp"])
    via[2] = max(via[2], float(walls[:, 2].max()) + args.via_clearance)
    wps2 = {"via": via, **wps}
    d2, lines2 = check(wps2)
    if d2 >= args.arm_margin:
        return wps2, ["direct route too close (" + "; ".join(lines) + ") -> via point above the walls"] + lines2
    return None, [f"arm would come {max(d, d2):.0f} mm from the cubicle (limit {args.arm_margin:g}): "
                  + "; ".join(lines2 if d2 > d else lines)]


def make_fence(robot, cell, args):
    """Live wall monitor for the virtual bin, or None (no bin.json / --no_bin)."""
    if args.no_bin:
        print("[FENCE] off (--no_bin)")
        return None
    from cube_pick.config import ROBOT_IP
    from . import workcell
    from .cli import build_config
    from .planner import ee_model
    bin_ = workcell.load()
    if bin_ is None:
        print("[FENCE] off: no bin.json (measure the board with:  python run.py bin)")
        return None
    gripper, _, params = build_config(args)
    walls = workcell.wall_points(bin_, args.bin_height)
    print(f"[FENCE] live wall monitor ON: walls {args.bin_height:g} mm, stops the robot within "
          f"{args.fence_margin:g} mm of a wall")
    return Fence(robot, ROBOT_IP, walls, ee_model(gripper, params.hover_radius), params.collision_tol,
                 cell.T_flange_gripper, margin=args.fence_margin, arm_walls=make_arm_walls(robot, args))


def go_home(cell, robot, vel):
    print(f"[MOVE] home: {np.round(HOME_POSE, 1).tolist()} (flange, vel {vel})")
    home_pos(robot, vel=vel)
    cell.wait_still()


def run(args, plan_fn):
    robot = connect_robot()
    st = robot.robot_state_pkg
    try:
        clear_faults(robot, ask=True)
    except RuntimeError as exc:
        raise SystemExit(f"[ABORT] {exc}")
    if args.execute:
        try:
            ensure_automatic(robot)   # SDK motion needs automatic mode
        except RuntimeError as exc:
            raise SystemExit(f"[ABORT] {exc}")
    else:
        print(f"[ROBOT] Mode: {'automatic' if st.robot_mode == 0 else 'manual'} "
              "(dry run: left as it is; --execute switches to automatic)")
    from cube_pick.config import TOOL_COORDS
    from .calibrate import load_cam_calib
    cam_rpy, cam_xyz, source = load_cam_calib(CAM_RPY_DEFAULT, TOOL_COORDS[args.cam_tool][:3])
    print(f"[CAM] camera calibration from {source}: rpy {np.round(cam_rpy, 2).tolist()}, "
          f"xyz {np.round(cam_xyz, 1).tolist()}")
    cell = Cell(robot, args.cam_tool, args.suction_tool, args.cam_yaw,
                None if args.use_tool_rotation else cam_rpy, cam_xyz=cam_xyz)
    home_joints(robot)  # raises if the home pose has no IK solution
    fence = make_fence(robot, cell, args)

    def mv(kind, pose, vel, label):
        """Move, watched by the live virtual-wall monitor (if a bin is set up)."""
        with (fence.watch(label) if fence else contextlib.nullcontext()):
            (cell.move_ptp if kind == "ptp" else cell.move)(pose, vel, label)

    def home(label):
        with (fence.watch(label) if fence else contextlib.nullcontext()):
            go_home(cell, robot, args.home_vel)
    if args.start_home:  # default: always measure from the same (home) pose
        flange = call("GetActualToolFlangePose", robot.GetActualToolFlangePose())
        dist = float(np.linalg.norm(np.subtract(flange[:3], HOME_POSE[:3])))
        if dist < 1.0:
            print("[HOME] Already at the home pose.")
        elif not args.execute:
            print(f"[DRY RUN] Robot is {dist:.0f} mm from home; --execute would move home first. "
                  "Measuring here instead.")
        elif input(f"Robot is {dist:.0f} mm from home. Type 'y' to move home ({args.home_vel:g}% speed) "
                   "and start: ").strip().lower() == "y":
            try:
                home("home (start)")
            except FenceStop as exc:
                print(f"[FENCE] STOPPED: {exc}. Jog the robot clear on the pendant.")
                return 1
            except KeyboardInterrupt:
                robot.StopMotion()
                print("\n[STOP] Ctrl+C -> StopMotion")
                return 130
        else:
            print("Cancelled.")
            return 0
    grasps = measure_and_plan(cell, args, plan_fn)
    if not grasps:
        print("[PLAN] No grasp found.")
        return 1

    chosen = None
    now = np.array(call("GetActualJointPosDegree", robot.GetActualJointPosDegree()))
    q_home = home_joints(robot)
    arm_walls = make_arm_walls(robot, args)
    for i, grasp in enumerate(grasps):
        options = []  # (largest joint change from now, grasp, waypoints)
        for g in equivalent_grasps(grasp):
            wps = waypoints(g, args)
            problem = linear_chain_problem(cell, [wps["pre-grasp"], wps["grasp"], wps["lift"]],
                                           ["pre-grasp", "grasp", "lift"])
            if problem:
                print(f"[IK] grasp #{i + 1}: {problem}")
                continue
            route, report = plan_route(cell, wps, now, q_home, arm_walls, args)
            if route is None:
                print(f"[ARM] grasp #{i + 1}{' (turned 180)' if g is not grasp else ''}: {report[0]}")
                continue
            options.append((float(np.max(np.abs(cell.reachable(wps["pre-grasp"]) - now))), g, route, report))
        if not options:
            print(f"[IK] grasp #{i + 1}: not usable; trying the next one")
            continue
        change, g, wps, arm_report = min(options, key=lambda o: o[0])  # least wrist/arm motion
        print(f"[IK] grasp #{i + 1}: largest joint change to pre-grasp {change:.0f} deg"
              f"{' (cups turned 180 deg: same grasp, less wrist motion)' if g is not grasp else ''}")
        chosen = (i, g, wps)
        print("[ARM] whole-arm distance to the cubicle per move: " + " | ".join(arm_report))
        break
    if chosen is None:
        print("[PLAN] None of the grasps is reachable from here.")
        return 1
    i, g, wps = chosen
    print(f"[CLEAR] highest obstacle near the path {g.terms.get('clear_height_mm', 0):.0f} mm above the parcel "
          f"-> hover / lift {approach_distance(g, args):.0f} mm above it (end effector checked against walls)")
    print(f"[PICK] grasp #{i + 1}: {g.n_objects} object(s), cups active {g.active.astype(int).tolist()}, "
          f"tilt {g.tilt_deg:.1f} deg")
    for name, pose in wps.items():
        print(f"   {name:9s} {np.round(pose, 1).tolist()}")
    place = None
    if args.place:
        place = {"place-above": list(args.place[:2]) + [args.place[2] + args.approach_mm] + list(args.place[3:]),
                 "place": list(args.place)}
        problem = linear_chain_problem(cell, [place["place-above"], place["place"]], ["place-above", "place"])
        if problem:
            raise SystemExit(f"[ABORT] place: {problem}")
        for name, pose in place.items():
            print(f"   {name:9s} {np.round(pose, 1).tolist()}")
    print(f"   {'home':9s} {np.round(HOME_POSE, 1).tolist()} (flange)"
          f"{'' if args.home else '  - skipped (--no_home)'}")
    if fence:
        planned = {name: fence.clearance(T_to_pose(pose_to_T(pose) @ np.linalg.inv(cell.T_flange_gripper)))
                   for name, pose in wps.items()}
        print("[FENCE] planned distance to the virtual walls: " +
              ", ".join(f"{k} {d:.0f} mm ({part})" for k, (d, part, _) in planned.items()))
    if not args.execute:
        print("[DRY RUN] Nothing moved. Add --execute (with --vacuum_do) to pick.")
        return 0

    vacuum = Vacuum(robot, args.vacuum_do, args.tool_io, len(g.active),
                    None if args.release_do < 0 else args.release_do, args.release_time)
    if input("Type 'y' to pick: ").strip().lower() != "y":
        print("Cancelled.")
        return 0
    try:
        vacuum.set(False)                                             # start: suction and release off
        vacuum.release_off()
        if "via" in wps:                                              # up and over the walls
            mv("ptp", wps["via"], VEL_TRAVEL, "via (above the walls)")
            mv("lin", wps["pre-grasp"], VEL_TRAVEL, "pre-grasp (hover)")
        else:
            mv("ptp", wps["pre-grasp"], VEL_TRAVEL, "pre-grasp (hover)")  # free-space travel
        vacuum.set(True, g.active)                                    # suction ON at the hover point
        mv("lin", wps["grasp"], VEL_DESCEND, "grasp")
        time.sleep(args.vacuum_wait)
        if args.vacuum_di is not None:
            ok = call("GetDI", robot.GetDI(args.vacuum_di, 0))
            if ok != 1:
                print("[VACUUM] No vacuum confirmed by the sensor input: releasing and backing off.")
                vacuum.release()
                mv("lin", wps["pre-grasp"], VEL_LIFT, "back off")
                if "via" in wps:
                    mv("lin", wps["via"], VEL_LIFT, "up to via (above the walls)")
                if args.home:
                    home("home")
                return 1
        mv("lin", wps["lift"], VEL_LIFT, "lift")
        if "via" in wps:
            mv("lin", wps["via"], VEL_LIFT, "up to via (above the walls)")
        if place:
            mv("ptp", place["place-above"], VEL_TRAVEL, "place-above")
            mv("lin", place["place"], VEL_DESCEND, "place")
            vacuum.release()                                          # suction off + release pulse
            mv("lin", place["place-above"], VEL_LIFT, "retreat")
            print("[PICK] Placed.")
            if args.home:
                home("home")
        elif args.home:
            home("home")                       # carry the parcel home
            vacuum.release()                                          # suction off + release pulse at home
            print("[PICK] Released the parcel at the home pose.")
        else:
            print("[PICK] Holding the parcel (suction ON): --no_home and no --place given.")
        print("[PICK] Done: one parcel. Run again for the next one.")
        return 0
    except FenceStop as exc:
        print(f"[FENCE] STOPPED: {exc}. Suction left as it was (a held parcel is not dropped). "
              "Jog the robot clear on the pendant.")
        return 1
    except KeyboardInterrupt:
        robot.StopMotion()
        print("\n[STOP] Ctrl+C -> StopMotion. Vacuum left as it was (switching it off would drop "
              "a held object).")
        return 130
