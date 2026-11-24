"""Fixed 'look at the table' home pose for the Fairino cobot (flange pose, tool 0)."""

import numpy as np

# Flange pose (tool 0) in the base frame: x y z (mm), rx ry rz (deg). Camera looks down at the table.
# (= camera tool 5 at [480.06, -711.48, 765.70, -178.62, -1.00, -4.30], as shown on the pendant
#  with tool 5 active, 2026-11-17)
HOME_POSE = [449.3558654785156, -656.0225830078125, 1032.5064086914062,
             161.18832397460938, 2.323822021484375, -147.6291046142578]


def home_joints(robot):
    """Joint solution for HOME_POSE, whatever tool is active on the pendant.

    The controller's IK solves for the ACTIVE tool, so the flange pose is first
    re-expressed as that tool's pose (otherwise MoveL fails with error 74).
    """
    from cube_pick.config import active_tool
    from cube_pick.robot.pick_cube import T_to_pose, pose_to_T
    T_tool = pose_to_T(active_tool(robot)[1])  # live offset of the active tool (GetTCPOffset)
    err, joints = robot.GetInverseKin(0, T_to_pose(pose_to_T(HOME_POSE) @ T_tool), -1)
    if err != 0 or joints is None:
        raise RuntimeError(f"No IK solution for the home pose (error {err})")
    return list(joints)


def home_pos(robot, vel=70):
    """Joint-space (PTP) move to HOME_POSE. A straight-line MoveL would fault (1/33) whenever
    home and the current pose use different joint configurations."""
    print("Current robot pos:", robot.GetActualTCPPose())
    err = robot.MoveJ(home_joints(robot), 0, 0, desc_pos=list(HOME_POSE), vel=vel)
    if err != 0:
        raise RuntimeError(f"MoveJ home failed with error code {err}")


def main(argv=None):
    """Move the robot to HOME_POSE:  python -m cube_pick.robot.home_pos [--vel 20] [--yes]"""
    import argparse
    import time

    from cube_pick.config import clear_faults, connect_robot, ensure_automatic
    from cube_pick.robot.pick_cube import call

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--vel", type=float, default=20.0, help="Speed in %% (default 20)")
    p.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    args = p.parse_args(argv)
    if not 0 < args.vel <= 100:
        p.error("--vel must be in (0, 100]")

    robot = connect_robot()
    time.sleep(0.5)  # let the status feed fill robot_state_pkg
    try:
        clear_faults(robot)
    except RuntimeError as exc:
        raise SystemExit(f"[ABORT] {exc}")
    flange = call("GetActualToolFlangePose", robot.GetActualToolFlangePose())
    dist = float(np.linalg.norm(np.subtract(flange[:3], HOME_POSE[:3])))
    print(f"[HOME] flange now {np.round(flange, 1).tolist()}")
    print(f"[HOME] home       {np.round(HOME_POSE, 1).tolist()}  ({dist:.0f} mm away)")
    if dist < 0.5:
        print("[HOME] Already at home.")
        return 0
    home_joints(robot)  # fail early if unreachable
    if not args.yes and input(f"Type 'y' to move home (joint move, {args.vel:g}% speed): ").strip().lower() != "y":
        print("Cancelled.")
        return 0
    try:
        ensure_automatic(robot)
    except RuntimeError as exc:
        raise SystemExit(f"[ABORT] {exc}")
    try:
        home_pos(robot, vel=args.vel)
        deadline = time.monotonic() + 120
        while call("GetRobotMotionDone", robot.GetRobotMotionDone()) != 1:
            if time.monotonic() > deadline:
                raise TimeoutError("Robot did not reach home in time")
            time.sleep(0.05)
    except KeyboardInterrupt:
        robot.StopMotion()
        print("\n[STOP] Ctrl+C -> StopMotion")
        return 130
    print("[HOME] At home.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
