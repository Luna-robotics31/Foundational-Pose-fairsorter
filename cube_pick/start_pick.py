#!/usr/bin/env python3
"""Start the whole cube pick with one command.

  1. Checks nothing else is holding the camera or the robot connection.
  2. Starts FoundationPose tracking (cube_pick.vision.live_tracker) in the background.
  3. Waits until the cube is tracked with a fresh, valid pose.
  4. Runs cube_pick.robot.pick_cube (dry run unless --execute is given).
  5. Stops the tracker cleanly, also on Ctrl+C or errors.

Usage (from the FoundationPose folder, project .venv active):
    python -m cube_pick                        # dry run: prints the plan, robot does not move
    python -m cube_pick --execute --hover_only
    python -m cube_pick --execute              # full pick (asks for 'y' first)
    python -m cube_pick --headless --execute   # no camera preview window

(`python cube_pick/start_pick.py ...` works too.)

Any option not listed under --help here is passed on to cube_pick.robot.pick_cube
(e.g. --check, --go_home, --height_source pose, --table_z 112, --cam_tool 5).
Tracker output goes to debug/live/run_live.log next to the pose file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if not __package__:  # run as a plain script: make `cube_pick` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cube_pick.config import FP_ROOT, MESH_FILE, POSE_JSON, TRACKER_LOG as LOG_FILE  # noqa: E402

TRACKER_MODULE = "cube_pick.vision.live_tracker"
PICK_MODULE = "cube_pick.robot.pick_cube"
DEFAULT_PICK_ARGS = ["--cam_tool", "5", "--gripper_tool", "1"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh_file", type=Path, default=MESH_FILE, help="Mesh for FoundationPose (metres)")
    p.add_argument("--headless", action="store_true", help="Run tracking without the preview window")
    p.add_argument("--track_timeout", type=float, default=90.0,
                   help="Seconds to wait for the first valid cube pose (model loading included)")
    p.add_argument("--keep_tracking", action="store_true",
                   help="Leave the tracker running after the pick")
    return p.parse_known_args()


def blocking_processes():
    """Other processes that would hold the camera or the robot's real-time port."""
    out = subprocess.run(["ps", "-eo", "pid=,stat=,args="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        pid, stat, cmd = line.split(None, 2)
        if int(pid) == os.getpid():
            continue
        if any(name in cmd for name in (TRACKER_MODULE, PICK_MODULE, "run_live_orbbec.py",
                                        "pick_cube_foundationpose.py", "python main.py")):
            found.append(f"  PID {pid} [{'PAUSED (Ctrl+Z)' if stat.startswith('T') else stat}] {cmd}")
    return found


def wait_for_tracking(proc, started_ns, timeout):
    print(f"[START] Waiting up to {timeout:.0f} s for the cube to be tracked "
          "(put it in view on the table)...")
    deadline, last_status = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Tracker exited with code {proc.returncode}; see {LOG_FILE}")
        try:
            record = json.loads(POSE_JSON.read_text())
        except (OSError, json.JSONDecodeError):
            record = {}
        status = record.get("status")
        if status and status != last_status:
            print(f"[TRACK] {status}")
            last_status = status
        if (record.get("valid") and (record.get("frame_received_unix_ns") or 0) > started_ns
                and record.get("quality", {}).get("depth_fit_fraction", 0) >= 0.75):
            print(f"[TRACK] Cube tracked, depth fit {record['quality']['depth_fit_fraction']:.0%}")
            return
        time.sleep(0.2)
    raise TimeoutError(f"No valid cube pose within {timeout:.0f} s; see {LOG_FILE}")


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    print("[STOP] Stopping the tracker ...")
    proc.send_signal(signal.SIGINT)  # its KeyboardInterrupt handler releases the camera
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def tail(path, lines=15):
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def main():
    args, pick_args = parse_args()
    blockers = blocking_processes()
    if blockers:
        print("[START] These processes would block the camera or the robot. Stop them first\n"
              "        (Ctrl+C/Q in their terminal, or `kill -9 <PID>` for PAUSED ones):")
        print("\n".join(blockers))
        return 1

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    live_cmd = [sys.executable, "-m", TRACKER_MODULE, "--mesh_file", str(args.mesh_file.resolve())]
    if args.headless:
        live_cmd.append("--headless")
    proc, started_ns = None, time.time_ns()
    try:
        with LOG_FILE.open("w") as log:
            print(f"[START] {' '.join(live_cmd[1:])}  (log: {LOG_FILE})")
            # Own session: the terminal's Ctrl+C reaches only the pick; we stop the tracker.
            proc = subprocess.Popen(live_cmd, cwd=FP_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            wait_for_tracking(proc, started_ns, args.track_timeout)

            pick_cmd = [sys.executable, "-m", PICK_MODULE, *DEFAULT_PICK_ARGS, *pick_args]
            print(f"[START] {' '.join(pick_cmd[1:])}\n")
            # Ctrl+C goes to the pick script, which stops the robot itself; the launcher
            # must not kill it mid-StopMotion, so it only waits (a Python handler, not
            # SIG_IGN, so the child still gets a normal KeyboardInterrupt).
            previous = signal.signal(signal.SIGINT, lambda *_: print("\n[START] Ctrl+C -> pick script"))
            try:
                pick = subprocess.Popen(pick_cmd, cwd=FP_ROOT)
                pick.wait()
            finally:
                signal.signal(signal.SIGINT, previous)
            result = pick
            print(f"\n[START] Pick script finished with code {result.returncode}")
            if args.keep_tracking:
                print(f"[START] Tracking left running (PID {proc.pid}); stop it with "
                      f"`kill -INT {proc.pid}`.")
                proc = None
            return result.returncode
    except KeyboardInterrupt:
        print("\n[START] Interrupted.")
        return 130
    except (RuntimeError, TimeoutError) as exc:
        print(f"[ERROR] {exc}\n--- last tracker log lines ---\n{tail(LOG_FILE)}")
        return 1
    finally:
        stop(proc)


if __name__ == "__main__":
    sys.exit(main())
