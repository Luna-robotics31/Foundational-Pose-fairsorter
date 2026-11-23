#!/usr/bin/env python3
"""Switch a Fairino digital output (e.g. the suction valve) on, check it, and switch it off.

Run from the suction_pick folder (or anywhere):
    python suction_activate.py                 # control-box DO1 on for 2 s, then off
    python suction_activate.py --do 2          # another output
    python suction_activate.py --tool          # tool-flange output (SetToolDO, id 0 or 1)
    python suction_activate.py --keep_on       # leave it on
The robot does not move.
"""
import argparse
import sys
import time
from pathlib import Path

# Make the FoundationPose folder importable, so the bundled Fairino SDK is found
# (cube_pick/robot/fairino_driver/Robot.py) - there is no 'fairino' package installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_pick.config import connect_robot  # noqa: E402


def output_state(robot, port, tool):
    """Current state (0/1) of a control-box or tool digital output, read from the robot."""
    if tool:
        err, bits = robot.GetToolDO()
    else:
        err, (high, low) = robot.GetDO()      # outputs 15-8 and 7-0
        bits = (high << 8) | low
    return None if err != 0 else (bits >> port) & 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--do", type=int, default=1, help="Output number (control box 0-15, tool 0-1); default 1")
    p.add_argument("--tool", action="store_true", help="Use the tool-flange outputs (SetToolDO)")
    p.add_argument("--seconds", type=float, default=2.0, help="How long to keep it on (default 2)")
    p.add_argument("--keep_on", action="store_true", help="Leave the output on")
    args = p.parse_args()
    limit = 1 if args.tool else 15
    if not 0 <= args.do <= limit:
        p.error(f"--do must be 0-{limit}")

    robot = connect_robot()                      # also waits for live status data
    name = f"{'tool' if args.tool else 'control box'} DO{args.do}"
    set_do = robot.SetToolDO if args.tool else robot.SetDO
    time.sleep(0.3)
    print(f"{name} before: {output_state(robot, args.do, args.tool)}")

    err = set_do(args.do, 1, 0, 0)               # id, status 1 = on, smooth 0, block 0
    print(f"{name} ON  -> error code {err} ({'OK' if err == 0 else 'FAILED'})")
    time.sleep(0.3)                              # let the status feed update
    print(f"{name} now: {output_state(robot, args.do, args.tool)}  (1 = on)")
    if err != 0 or args.keep_on:
        return 0 if err == 0 else 1
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        err = set_do(args.do, 0, 0, 0)
        print(f"{name} OFF -> error code {err} ({'OK' if err == 0 else 'FAILED'})")
        time.sleep(0.3)
        print(f"{name} now: {output_state(robot, args.do, args.tool)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
