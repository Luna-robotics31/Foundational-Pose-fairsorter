"""Where the suction pick keeps its files (all relative to the suction_pick folder)."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]     # .../suction_pick
DATA_DIR = PACKAGE_DIR / "data"                        # saved setup for this robot / table
OUTPUT_DIR = PACKAGE_DIR / "output"                    # one folder per run (git-ignored)

BIN_FILE = DATA_DIR / "bin.json"                       # table outline -> virtual bin (python run.py bin)
CALIB_FILE = DATA_DIR / "camera_calibration.json"      # camera on the arm (python run.py calibrate)
RVIZ_CONFIG = OUTPUT_DIR / "sim.rviz"                  # written by python run.py sim
RVIZ_LOG = OUTPUT_DIR / "rviz.log"
