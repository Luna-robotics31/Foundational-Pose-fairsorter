"""Shared paths and constants for the cube-pick pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
FP_ROOT = PACKAGE_DIR.parent  # FoundationPose checkout: estimater.py, Utils.py, mycpp/, weights/

# Assets
ASSETS_DIR = PACKAGE_DIR / "assets"
MESH_FILE = ASSETS_DIR / "cube_recessed.obj"   # cube model, metres
DETECTOR_WEIGHTS = ASSETS_DIR / "kaushal.pt"   # YOLO detector, class "white_cube"

# Runtime output (git-ignored)
LIVE_DIR = FP_ROOT / "debug" / "live"
POSE_JSON = LIVE_DIR / "latest_pose.json"      # written by the tracker, read by the pick
TRACKER_LOG = LIVE_DIR / "run_live.log"

# Robot
FAIRINO_DRIVER_DIR = PACKAGE_DIR / "robot" / "fairino_driver"
ROBOT_IP = "192.168.58.2"

# Tool coordinates (flange -> tool TCP) exactly as set on the pendant: x y z (mm), rx ry rz (deg).
# Kept here because this controller (software v3.8.5) has no GetToolCoordWithID; only the
# ACTIVE tool's offset can be read (GetTCPOffset). Update these if a tool is re-taught.
CAM_TOOL = 5        # camera: centre of the Gemini 2 RGB lens (rotation replaced by CAM_RPY_DEFAULT)
SUCTION_TOOL = 1    # suction cup TCP (replaces the old finger gripper): between the cups, lip plane
TOOL_COORDS = {
    0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                       # bare flange
    SUCTION_TOOL: [2.333, 0.564, 127.237, 0.0, 0.0, 0.0],    # read from the controller 2026-11-16
    CAM_TOOL: [5.933, -77.174, 30.342, -15.248, 10.272, -144.596],   # re-taught on the pendant 2026-11-17
    #          (was [41.02, -56.34, 17.39, -163.37, 18.18, 2.97])
}


def tool_coord(tool_id: int) -> list:
    """Flange -> TCP offset of a tool from TOOL_COORDS (no controller query)."""
    if tool_id not in TOOL_COORDS:
        raise KeyError(f"Tool {tool_id} is not in TOOL_COORDS (cube_pick/config.py); add its "
                       "x y z rx ry rz from the pendant.")
    return list(TOOL_COORDS[tool_id])


def active_tool(robot) -> tuple[int, list]:
    """(id, flange -> TCP offset) of the tool ACTIVE on the pendant, read from the controller.

    The controller's IK and its TCP readout use this tool, so it is read live (GetTCPOffset)
    rather than taken from TOOL_COORDS.
    """
    err, tool_id = robot.GetActualTCPNum()
    if err != 0:
        raise RuntimeError(f"GetActualTCPNum failed with error code {err}")
    err, offset = robot.GetTCPOffset()
    if err != 0:
        raise RuntimeError(f"GetTCPOffset failed with error code {err}")
    return int(tool_id), [float(v) for v in offset]


def add_to_path(path: Path) -> None:
    """Put a directory first on sys.path (once)."""
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def connect_robot(timeout_s: float = 5.0):
    """Return a Fairino RPC connection using the bundled SDK, once live status data arrives.

    Until the first status packet arrives, robot.robot_state_pkg is the empty ctypes CLASS
    (its fields read as '<Field ...>', which looks like a fault and has no real values).
    """
    import time

    add_to_path(FAIRINO_DRIVER_DIR)
    import Robot
    robot = Robot.RPC(ROBOT_IP)
    deadline = time.monotonic() + timeout_s
    while isinstance(robot.robot_state_pkg, type):
        if time.monotonic() > deadline:
            raise RuntimeError(f"No status data from the robot at {ROBOT_IP} within {timeout_s:g} s "
                               "(check the Ethernet link and that the controller is on).")
        time.sleep(0.05)
    time.sleep(0.2)  # let a full packet populate every field
    return robot


def ensure_automatic(robot, timeout_s: float = 5.0) -> None:
    """Switch the robot to AUTOMATIC mode (needed for SDK motion) and confirm it.

    The SDK's own robot.Mode(0) retries forever if the controller does not answer (it froze
    the pick once), so the command goes over a fresh XML-RPC connection with a timeout and
    the result is confirmed from the robot's status feed. Raises RuntimeError on failure.
    """
    import time
    import xmlrpc.client

    state = robot.robot_state_pkg
    if state.robot_mode == 0:
        print("[ROBOT] Mode: automatic")
        return
    if state.robot_state == 4:
        raise RuntimeError("Robot is in drag-teach mode; leave it on the pendant first.")

    class _Timeout(xmlrpc.client.Transport):
        def make_connection(self, host):
            conn = super().make_connection(host)
            conn.timeout = timeout_s
            return conn
    print("return to the original pose after the suction gripper has opened or closed")

    print("[ROBOT] Mode: manual -> switching to AUTOMATIC")
    try:
        err = xmlrpc.client.ServerProxy(f"http://{ROBOT_IP}:20003", transport=_Timeout()).Mode(0)
    except Exception as exc:
        raise RuntimeError(f"Mode(0) got no answer ({exc}). Switch to automatic on the pendant.") from exc
    if err != 0:
        raise RuntimeError(f"Mode(0) failed with error code {err}. Switch to automatic on the pendant "
                           "(check that no error is active and the pendant is not holding control).")
    deadline = time.monotonic() + timeout_s
    while robot.robot_state_pkg.robot_mode != 0:
        if time.monotonic() > deadline:
            raise RuntimeError("Robot did not report automatic mode after Mode(0). Switch it on the pendant.")
        time.sleep(0.1)
    print("[ROBOT] Mode: automatic (switched)")


def clear_faults(robot, ask: bool = True) -> None:
    """If the controller reports a fault, show it and (after 'r') reset it with ResetAllError.

    Resettable faults such as 1/33 ("next command joint configuration changed") block all
    motion until reset. Raises RuntimeError if a fault stays active.
    """
    import time

    st = robot.robot_state_pkg
    if not (st.main_code or st.sub_code):
        return
    print(f"[ROBOT] Active fault: main {st.main_code} / sub {st.sub_code} "
          "(see the FAIRINO manual, appendix 'Motion controller errors')")
    if ask and input("Type 'r' to reset the fault and continue: ").strip().lower() != "r":
        raise RuntimeError(f"Fault {st.main_code}/{st.sub_code} not reset.")
    err = robot.ResetAllError()
    deadline = time.monotonic() + 3.0
    while (st.main_code or st.sub_code) and time.monotonic() < deadline:
        time.sleep(0.1)
    if st.main_code or st.sub_code:
        raise RuntimeError(f"Fault {st.main_code}/{st.sub_code} is still active after ResetAllError "
                           f"(code {err}); check the pendant.")
    print("[ROBOT] Fault reset.")
