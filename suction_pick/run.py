#!/usr/bin/env python3
"""The ONLY entry point of the suction pick. Run it from this folder:

    python run.py demo         synthetic test scene (no robot, no camera)
    python run.py io           test a suction output (DO0 / DO1)
    python run.py plan         camera only: find parcels and plan, no robot
    python run.py bin          measure the table -> virtual bin (robot does not move)
    python run.py calibrate    camera calibration (robot moves)
    python run.py pick         dry run on the robot (nothing moves); add --execute to pick
    python run.py sim          RViz2 simulation of the last dry run

`python run.py <command> --help` lists the options. Everything else lives in the sub-folders.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Import the package from the parent folder (FoundationPose), not this folder's files as
# top-level modules (e.g. robot.py, config.py would otherwise shadow other names).
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(HERE.parent))

from suction_pick.app.cli import main  # noqa: E402
import rich 
rich.print("conig file for the cobot coordinates will be saved in grasps.json")
sys.exit(main())
    