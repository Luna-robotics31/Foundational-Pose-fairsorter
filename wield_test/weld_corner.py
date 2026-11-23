#!/usr/bin/env python3
"""Weld-torch test: L-shaped seam with a smooth corner on a Fairino cobot (no arc is fired).

The seam is p1 -> p2 (line 1), a ~90 deg corner, p3 -> p4 (line 2), taught with MoveL.
Four ways to drive the corner, so they can be compared on the real robot:

  sharp   MoveL p2, p3, p4 with stops (blendR -1)       the original, vibrating corner
  blend   MoveL p2, p3 with the controller's blendR      controller rounds both small corners
  bezier  line -> quintic Bezier corner -> line, sent as a chain of short MoveL with a
          small blendR (recommended first try)
  spline  the same Bezier path sent as one Fairino Cartesian spline (NewSpline*)

Usage (from the FoundationPose folder, spark_src .venv active):
  python weld_test/weld_corner.py                              # dry run: report + plot, no robot
  python weld_test/weld_corner.py --tool 1 --check             # connect, check IK on every point
  python weld_test/weld_corner.py --tool 1 --execute --method bezier
  python weld_test/weld_corner.py --tool 1 --execute --method sharp   # compare with the original

--tool/--user must be the tool/workpiece numbers that were ACTIVE when p1..p4 were recorded
(the pose readout is for the active TCP). The script refuses to move if the robot's active
tool/workpiece differ. The robot must be in AUTOMATIC mode (set it on the pendant).

STOP while it moves: press ENTER (or Ctrl+C) in this terminal -> StopMotion is sent at once
on its own connection. A robot fault (e.g. its collision detection or the e-stop) also
stops the script. After a stop the robot is NOT moved again; jog it clear on the pendant.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import threading
import time
import xmlrpc.client
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from corner_path import Seam, rot_of

HERE = Path(__file__).resolve().parent
DRIVER_DIR = HERE.parent / "cube_pick/robot/fairino_driver"
OUT_DIR = HERE / "output"
ROBOT_IP = "192.168.58.2"
J5_SINGULAR_DEG = 10.0      # warn when the wrist (J5) is this close to 0 deg


@dataclass
class Move:
    pose: list[float]
    blend: float     # MoveL blendR in mm; -1 = stop at the point
    label: str


# ── plan ────────────────────────────────────────────────────────────────
def build_program(seam: Seam, args) -> list[Move]:
    """Motion after the robot is at p1."""
    p = seam.taught
    if args.method == "sharp":
        return [Move(p[1], -1, "p2"), Move(p[2], -1, "p3"), Move(p[3], -1, "p4")]
    if args.method == "blend":
        limit = 0.5 * min(seam.len1, float(np.linalg.norm(seam.x[2] - seam.x[1])), seam.len2)
        if not 0 < args.ctrl_blend_mm <= limit:
            sys.exit(f"--ctrl_blend_mm must be in (0, {limit:.1f}] (half the shortest segment).")
        return [Move(p[1], args.ctrl_blend_mm, "p2"), Move(p[2], args.ctrl_blend_mm, "p3"),
                Move(p[3], -1, "p4")]
    if args.method == "bezier":
        corner = seam.corner_samples(args.step_mm)        # A ... B
        blend = args.blend_mm if args.blend_mm is not None else 0.4 * args.step_mm
        spacing = min(b.s - a.s for a, b in zip(corner[:-1], corner[1:]))
        if not 0 < blend < 0.5 * spacing:
            sys.exit(f"--blend_mm must be in (0, {0.5 * spacing:.2f}) for --step_mm {args.step_mm}.")
        moves = [Move(c.pose, blend, f"corner {i}/{len(corner) - 1}") for i, c in enumerate(corner)]
        moves[0].label, moves[-1].label = "A (corner start)", "B (corner end)"
        return moves + [Move(p[3], -1, "p4")]
    # spline: every point of the path after p1
    path = seam.dense_path(args.step_mm, args.line_step_mm)[1:]
    return [Move(s.pose, 0.0, f"{s.part} {s.s:.1f}mm") for s in path]


def approach_pose(pose, back_mm):
    """Pose moved back along the torch axis (tool Z) by back_mm."""
    z = rot_of(pose).apply([0, 0, 1])
    return [*(np.asarray(pose[:3]) - back_mm * z), *pose[3:]]


def report(seam: Seam, args, program: list[Move]):
    s = seam.summary(args.step_mm)
    print(f"[SEAM] turn {s['turn_deg']:.1f} deg | lines miss by {s['line_gap_mm']:.2f} mm | "
          f"corner at {np.round(s['corner_xyz'], 1).tolist()}")
    print(f"[SEAM] taught p2 is {s['p2_before_corner_mm']:.1f} mm before the corner, "
          f"p3 {s['p3_after_corner_mm']:.1f} mm after it; torch turns {s['torch_turn_deg']:.1f} deg")
    print(f"[CURVE] corner size {seam.d:.1f} mm -> curve {s['corner_len_mm']:.1f} mm long, "
          f"min radius {s['min_radius_mm']:.1f} mm, passes {s['closest_to_corner_mm']:.2f} mm "
          f"from the sharp corner")
    print(f"[CURVE] curvature at the ends {s['end_curvature'][0]:.1e} / {s['end_curvature'][1]:.1e} "
          f"1/mm (0 = no jerk entering/leaving); peak torch turn "
          f"{s['max_torch_turn_deg_per_mm']:.1f} deg/mm")
    print(f"[PATH] {s['total_len_mm']:.1f} mm total | method '{args.method}': {len(program)} moves "
          "after p1")


def save_outputs(seam: Seam, args, program: list[Move]):
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / f"program_{args.method}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "x", "y", "z", "rx", "ry", "rz", "blendR"])
        w.writerow(["p1 (start)", *np.round(seam.taught[0], 4), -1])
        for m in program:
            w.writerow([m.label, *np.round(m.pose, 4), m.blend])
    print(f"[OUT] {csv_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[OUT] matplotlib not installed; no plot.")
        return
    dense = seam.dense_path(0.25, 1.0)
    xyz = np.array([d.pose[:3] for d in dense])
    taught = np.array(seam.taught)[:, :3]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    a = ax[0]
    a.plot(taught[:, 0], taught[:, 1], "o--", color="0.6", label="taught p1-p2-p3-p4 (sharp)")
    for i, t in enumerate(taught):
        a.annotate(f"p{i + 1}", t[:2], textcoords="offset points", xytext=(6, 6))
    a.plot(*np.array([seam.A, seam.corner, seam.B])[:, :2].T, ":", color="0.4", label="extended lines")
    a.plot(xyz[:, 0], xyz[:, 1], "-", color="C0", lw=2, label="planned path")
    prog = np.array([m.pose[:3] for m in program])
    a.plot(prog[:, 0], prog[:, 1], ".", color="C3", ms=5, label=f"'{args.method}' targets")
    for d in dense[::12]:
        z = rot_of(d.pose).apply([0, 0, 1])
        a.arrow(d.pose[0] - 12 * z[0], d.pose[1] - 12 * z[1], 12 * z[0], 12 * z[1],
                color="C2", width=0.15, length_includes_head=True)
    a.set_aspect("equal"); a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    a.set_title("Top view (green: torch axis, tip at the path)"); a.legend(fontsize=8)
    s_c, k = seam.curvature()
    ax[1].plot(s_c, k, color="C0")
    ax[1].set_xlabel("arc length along the corner (mm)"); ax[1].set_ylabel("curvature (1/mm)")
    ax[1].set_title("Corner curvature: starts and ends at 0")
    s = np.array([d.s for d in dense])
    ang = [0.0]
    for p, q in zip(dense[:-1], dense[1:]):
        ang.append(ang[-1] + np.degrees((rot_of(p.pose).inv() * rot_of(q.pose)).magnitude()))
    ax[2].plot(s, ang, color="C1")
    ax[2].axvspan(seam.line1_len, seam.line1_len + seam.corner_len, color="C0", alpha=0.1,
                  label="corner")
    ax[2].set_xlabel("path length from p1 (mm)"); ax[2].set_ylabel("torch rotation so far (deg)")
    ax[2].set_title("Torch orientation along the seam"); ax[2].legend()
    fig.tight_layout()
    png = OUT_DIR / "corner_path.png"
    fig.savefig(png, dpi=110)
    print(f"[OUT] {png}")


# ── robot ───────────────────────────────────────────────────────────────
def call(name, result):
    """Fairino calls return err or (err, value...); raise on a non-zero error code."""
    if isinstance(result, int):
        err, value = result, None
    else:
        err, value = result[0], (result[1] if len(result) > 1 else None)
    if err != 0:
        raise RuntimeError(f"{name} failed with error code {err}")
    return value


class Stopped(Exception):
    """The run was stopped (Enter, Ctrl+C or a robot fault)."""


class _TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = 2.0
        return conn


class StopSwitch:
    """Stop the robot from the terminal while it moves: ENTER or Ctrl+C.

    StopMotion goes over a NEW XML-RPC connection every time: the SDK's own connection
    may be stuck in the middle of a (blocking) MoveL call when the stop is needed.
    """

    def __init__(self, ip):
        self.url = f"http://{ip}:20003"
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._old_sigint = None

    def start(self):
        self._old_sigint = signal.signal(signal.SIGINT, self._on_sigint)
        threading.Thread(target=self._listen, daemon=True).start()
        print(">>> Press ENTER (or Ctrl+C) at any time to STOP the robot. <<<")

    def close(self):
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)
            self._old_sigint = None

    def _listen(self):
        if sys.stdin.readline():  # '' = stdin closed, nothing to listen to
            self.stop("ENTER pressed")

    def _on_sigint(self, *_):
        self.stop("Ctrl+C")
        raise KeyboardInterrupt

    def stop(self, reason):
        with self._lock:
            first = not self.event.is_set()
            self.event.set()
        if first:
            print(f"\n[STOP] {reason} -> StopMotion", flush=True)
        for attempt in range(3):
            try:
                err = xmlrpc.client.ServerProxy(self.url, transport=_TimeoutTransport()).StopMotion()
                if first:
                    print(f"[STOP] Robot stopped (StopMotion code {err}). Jog it clear on the pendant.",
                          flush=True)
                return
            except Exception as exc:  # network trouble: retry, then tell the operator
                last = exc
        print(f"[STOP] StopMotion FAILED ({last}). PRESS THE E-STOP.", flush=True)

    def check(self, robot):
        """Raise Stopped if the operator stopped the run or the robot reports a fault."""
        st = robot.robot_state_pkg
        if not self.event.is_set() and (st.main_code or st.sub_code or st.EmergencyStop):
            self.stop(f"robot fault {st.main_code}/{st.sub_code}"
                      f"{' (E-STOP)' if st.EmergencyStop else ' (collision detection?)'}")
        if self.event.is_set():
            raise Stopped()


def connect(ip):
    sys.path.insert(0, str(DRIVER_DIR))
    import Robot
    robot = Robot.RPC(ip)
    time.sleep(0.5)  # let the state-feedback thread fill robot_state_pkg
    return robot


def check_robot_state(robot, args, need_auto):
    st = robot.robot_state_pkg
    if st.tool != args.tool or st.user != args.user:
        sys.exit(f"[ABORT] Active tool/workpiece on the robot is {st.tool}/{st.user}, but "
                 f"--tool/--user is {args.tool}/{args.user}. Select the torch tool that was used "
                 "to record p1..p4 on the pendant (IK uses the active tool).")
    if st.main_code or st.sub_code:
        sys.exit(f"[ABORT] Robot reports fault {st.main_code}/{st.sub_code}; clear it first.")
    if need_auto and st.robot_mode != 0:
        sys.exit("[ABORT] Robot is in MANUAL mode. Switch it to AUTOMATIC on the pendant.")


def solve_ik(robot, poses, labels, start_joints, max_step_deg):
    """IK for each pose, each referenced to the previous solution (one continuous branch)."""
    joints, ref, problems = [], list(start_joints), []
    for i, (pose, label) in enumerate(zip(poses, labels)):
        err, q = robot.GetInverseKinRef(0, pose, ref)
        if err != 0 or q is None:
            raise RuntimeError(f"No IK solution for {label} {np.round(pose, 1).tolist()} (error {err})")
        q = list(map(float, q))
        if i > 0:  # the first step (current pose -> approach) is a MoveJ, not a path step
            turn = np.degrees((rot_of(poses[i - 1]).inv() * rot_of(pose)).magnitude())
            step = float(np.max(np.abs(np.subtract(q, ref))))
            if step > max_step_deg + turn:
                problems.append(f"{label}: joints jump {step:.1f} deg (torch turns {turn:.1f} deg)")
        if abs(q[4]) < J5_SINGULAR_DEG:
            problems.append(f"{label}: J5 = {q[4]:.1f} deg (near wrist singularity)")
        joints.append(q)
        ref = q
    return joints, problems


def wait_at(robot, target, timeout_s, switch: StopSwitch, tol_mm=0.5):
    """Wait until the robot reports motion done AND the TCP is at `target` (so a queued
    motion that has not started yet is not mistaken for a finished one)."""
    deadline = time.monotonic() + timeout_s
    while True:
        switch.check(robot)
        done = call("GetRobotMotionDone", robot.GetRobotMotionDone()) == 1
        tcp = call("GetActualTCPPose", robot.GetActualTCPPose())
        if done and np.linalg.norm(np.subtract(tcp[:3], target[:3])) < tol_mm:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Robot did not reach {np.round(target[:3], 1).tolist()} in time "
                               f"(TCP at {np.round(tcp[:3], 1).tolist()})")
        time.sleep(0.05)


def run_robot(seam: Seam, args, program: list[Move]):
    robot = connect(args.ip)
    check_robot_state(robot, args, need_auto=args.execute)
    current = call("GetActualJointPosDegree", robot.GetActualJointPosDegree())
    tcp = call("GetActualTCPPose", robot.GetActualTCPPose())

    p1 = seam.taught[0]
    p4 = seam.taught[3]
    app_in, app_out = approach_pose(p1, args.approach_mm), approach_pose(p4, args.approach_mm)
    poses = [app_in, p1] + [m.pose for m in program] + [app_out]
    labels = ["approach", "p1"] + [m.label for m in program] + ["retract"]
    joints, problems = solve_ik(robot, poses, labels, current, args.max_joint_step)
    print(f"[IK] {len(poses)} poses solved on one branch. Current TCP is "
          f"{np.linalg.norm(np.subtract(tcp[:3], app_in[:3])):.0f} mm from the approach point.")
    j = np.array(joints)
    print(f"[IK] joint ranges along the path (deg): "
          + ", ".join(f"J{i + 1} {j[:, i].min():.0f}..{j[:, i].max():.0f}" for i in range(6)))
    if problems:
        print("[IK] PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit("[ABORT] Fix the points/corner size (or raise --max_joint_step if a jump is "
                 "expected) before moving.")
    if not args.execute:
        print("[CHECK] OK. Nothing moved. Add --execute to run it.")
        return 0

    print(f"\nPlan: MoveJ to {args.approach_mm:.0f} mm above p1 -> MoveL to p1 -> '{args.method}' "
          f"seam at vel {args.vel}% -> retract {args.approach_mm:.0f} mm. NO welding arc.")
    if input("Type 'y' to move the robot: ").strip().lower() != "y":
        print("Cancelled.")
        return 0

    j_app, j_p1, j_prog, j_out = joints[0], joints[1], joints[2:-1], joints[-1]
    switch = StopSwitch(args.ip)
    switch.start()
    fir_on = False

    def send(name, fn, *a, **k):
        """Send one robot command, unless the run has been stopped."""
        switch.check(robot)
        call(name, fn(*a, **k))
        switch.check(robot)  # a blocking MoveL may return because of the stop

    try:
        send("MoveJ approach", robot.MoveJ, j_app, args.tool, args.user, desc_pos=app_in,
             vel=args.approach_vel)
        wait_at(robot, app_in, 120, switch)
        send("MoveL p1", robot.MoveL, p1, args.tool, args.user, joint_pos=j_p1,
             vel=args.approach_vel, blendR=-1)
        wait_at(robot, p1, 60, switch)
        if args.fir:
            send("LinArcFIRPlanningStart", robot.LinArcFIRPlanningStart, *args.fir)
            fir_on = True
        t0 = time.monotonic()
        if args.method == "spline":
            send("NewSplineStart", robot.NewSplineStart, 1, args.spline_avg_time)
            for i, (m, q) in enumerate(zip(program, j_prog)):
                send(f"NewSplinePoint {m.label}", robot.NewSplinePoint, m.pose, args.tool,
                     args.user, int(i == len(program) - 1), joint_pos=q, ovl=args.vel, blendR=0.0)
            send("NewSplineEnd", robot.NewSplineEnd)
        else:
            for m, q in zip(program, j_prog):
                send(f"MoveL {m.label}", robot.MoveL, m.pose, args.tool, args.user, joint_pos=q,
                     vel=args.vel, blendR=m.blend, blendMode=args.blend_mode)
        wait_at(robot, p4, 300, switch)
        elapsed = time.monotonic() - t0
        print(f"[RUN] Seam done in {elapsed:.1f} s "
              f"({seam.summary(args.step_mm)['total_len_mm'] / elapsed:.1f} mm/s average, "
              "including command upload).")
        if fir_on:
            send("LinArcFIRPlanningEnd", robot.LinArcFIRPlanningEnd)
            fir_on = False
        send("MoveL retract", robot.MoveL, app_out, args.tool, args.user, joint_pos=j_out,
             vel=args.approach_vel, blendR=-1)
        wait_at(robot, app_out, 60, switch)
        print("[RUN] Done.")
        return 0
    except (Stopped, KeyboardInterrupt):
        switch.stop("stopped")  # no-op message if already stopped; resends StopMotion
        print("[STOP] Run aborted. The robot will not move again from this script.")
        return 130
    except Exception as exc:
        switch.stop(f"error: {exc}")
        if isinstance(exc, RuntimeError) and switch.event.is_set():
            print("[STOP] Run aborted.")
            return 1
        raise
    finally:
        switch.close()
        if fir_on:
            try:
                robot.LinArcFIRPlanningEnd()
            except Exception as exc:
                print(f"[WARN] Could not switch the FIR filter off ({exc}); it stays on until "
                      "the controller is restarted or LinArcFIRPlanningEnd is called.")


# ── main ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=Path, default=HERE / "points.json", help="JSON with p1..p4")
    p.add_argument("--method", choices=("sharp", "blend", "bezier", "spline"), default="bezier")
    p.add_argument("--corner_mm", type=float, default=10.0,
                   help="Corner size: the curve starts/ends this far from the corner along each "
                        "line (bigger = smoother, but cuts the corner more)")
    p.add_argument("--step_mm", type=float, default=1.0, help="Point spacing along the corner curve")
    p.add_argument("--line_step_mm", type=float, default=5.0,
                   help="spline only: point spacing along the straight lines")
    p.add_argument("--blend_mm", type=float, help="bezier only: MoveL blendR between corner points "
                                                    "(default 0.4 x step_mm)")
    p.add_argument("--ctrl_blend_mm", type=float, default=5.0, help="blend only: blendR at p2 and p3")
    p.add_argument("--blend_mode", type=int, choices=(0, 1), default=0,
                   help="MoveL blendMode: 0 inscribed (default), 1 corner")
    p.add_argument("--vel", type=float, default=5.0,
                   help="Seam speed in %% (MoveL vel / spline ovl); start low")
    p.add_argument("--approach_mm", type=float, default=30.0, help="Approach/retract along the torch axis")
    p.add_argument("--approach_vel", type=float, default=10.0, help="Approach/retract speed in %%")
    p.add_argument("--spline_avg_time", type=int, default=2000, help="spline only: NewSplineStart averageTime (ms)")
    p.add_argument("--fir", type=float, nargs=4, metavar=("ACC_LIN", "ACC_DEG", "JERK_LIN", "JERK_DEG"),
                   help="Enable the controller's LIN/ARC FIR jerk filter with these limits "
                        "(mm/s2, deg/s2, mm/s3, deg/s3); check the ranges in your Fairino manual")
    p.add_argument("--max_joint_step", type=float, default=30.0,
                   help="Abort if a joint moves more than this (+ the torch turn) between two points")
    p.add_argument("--tool", type=int, help="Torch tool number (active when p1..p4 were recorded)")
    p.add_argument("--user", type=int, default=0, help="Workpiece frame number (active when recorded)")
    p.add_argument("--ip", default=ROBOT_IP)
    p.add_argument("--check", action="store_true", help="Connect and check IK on every point; no motion")
    p.add_argument("--execute", action="store_true", help="Move the robot (asks for 'y' first)")
    args = p.parse_args()
    if (args.check or args.execute) and args.tool is None:
        p.error("--tool is required with --check/--execute")
    if not (0 < args.vel <= 100 and 0 < args.approach_vel <= 100):
        p.error("--vel and --approach_vel must be in (0, 100]")
    if not (0.1 <= args.step_mm <= 10 and 0.5 <= args.line_step_mm <= 50):
        p.error("--step_mm must be in [0.1, 10] and --line_step_mm in [0.5, 50]")
    if args.approach_mm < 0:
        p.error("--approach_mm must be >= 0")
    return args


def main():
    args = parse_args()
    pts = json.loads(args.points.read_text())
    try:
        seam = Seam(pts["p1"], pts["p2"], pts["p3"], pts["p4"], args.corner_mm)
    except ValueError as exc:
        sys.exit(f"[ABORT] {exc}")
    program = build_program(seam, args)
    report(seam, args, program)
    save_outputs(seam, args, program)
    if not (args.check or args.execute):
        print("[DRY RUN] No robot connection. Use --tool N --check, then --execute.")
        return 0
    return run_robot(seam, args, program)


if __name__ == "__main__":
    sys.exit(main())
