#!/usr/bin/env python3
"""Run suction_pick from inside this folder:  python run.py demo | plan | pick ...

Same as `python -m suction_pick ...` from the FoundationPose folder.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Import the package from the parent folder (FoundationPose), not this folder's files as
# top-level modules (e.g. robot.py, config.py would otherwise shadow other names).
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(HERE.parent))

from suction_pick.cli import main  # noqa: E402
import rich 
rich.print("conig file for the cobot coordinates will be saved in grasps.json")
sys.exit(main())
    