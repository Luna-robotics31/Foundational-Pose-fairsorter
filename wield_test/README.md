# Weld corner test: smooth Bezier corner on the Fairino cobot

Moves a welding torch along an L-shaped seam (p1 → p2, ~90° corner, p3 → p4) and replaces
the sharp corner with a smooth curve, to stop the arm vibrating in the corner.
**No welding arc is fired:** the script never calls `ARCStart`.

```
weld_test/
├── weld_corner.py   run it: dry run / --check / --execute, 4 methods to compare
├── corner_path.py   geometry only (Bezier corner, orientation, sampling, checks)
├── points.json      the taught points p1..p4
└── output/          program_<method>.csv + corner_path.png (written on every run)
```

## Why it vibrates and what the script does

Between p2 and p3 the torch changes direction by 86° and rotates by 86°, all within 13 mm.
Driven as separate MoveL moves, the robot either stops there or takes a sudden jerk.

The script extends both straight lines to the corner where they meet, then replaces the
corner with a **5th-order Bezier curve** whose first three control points lie on line 1 and
last three on line 2:

- The curve leaves and rejoins each line in exactly the same direction, so there's no kink.
- Its curvature is **zero** where it meets the lines, so the sideways acceleration doesn't
  jump. A circular arc (MoveC) or a controller blend does have that jump.
- The control points are placed to keep the change in curvature, and so the jerk, as small
  as possible.
- The torch turns its 86° with an S-shaped (smootherstep) profile: the rotation starts and
  ends at zero speed and zero acceleration.

`--corner_mm` is how far from the corner the curve starts along each line. For these points:

| corner_mm | tightest radius | curve passes the sharp corner at | torch turn rate |
|---|---|---|---|
| 5  | 3.9 mm  | 1.5 mm | 18.5 °/mm |
| 10 (default) | 7.3 mm | 3.0 mm | 9.3 °/mm |
| 20 | 14.4 mm | 5.9 mm | 4.7 °/mm |

A bigger corner is smoother, but the torch cuts more of the corner.

## Methods (`--method`)

| method | how the corner is driven |
|---|---|
| `sharp`  | MoveL p2, p3, p4 with stops: your original motion, the baseline |
| `blend`  | MoveL p2, p3 with the controller's `blendR` (`--ctrl_blend_mm`, max ≈ 6.7 mm here) |
| `bezier` | line → Bezier corner as a chain of short MoveL (1 mm apart, `blendR` 0.4 mm) → line. **Try this first.** |
| `spline` | the same Bezier path sent as one Fairino Cartesian spline (`NewSplineStart/Point/End`) |

Optionally add `--fir ACC_LIN ACC_DEG JERK_LIN JERK_DEG` to switch on the controller's
LIN/ARC jerk filter (`LinArcFIRPlanningStart`) during the seam. The script doesn't guess
these limits; take them from your Fairino manual.

Every robot command uses joint solutions the script computes beforehand along one
continuous IK branch, so the controller can't switch arm configuration halfway through.

## How to run

```bash
source /home/shashank/Projects/spark_src/.venv/bin/activate
cd /home/shashank/Projects/FoundationPose

python weld_test/weld_corner.py                                  # 1. dry run, look at output/corner_path.png
python weld_test/weld_corner.py --tool N --check                 # 2. robot connected: IK check, no motion
python weld_test/weld_corner.py --tool N --execute --method sharp    # 3. baseline
python weld_test/weld_corner.py --tool N --execute --method bezier   # 4. smooth corner
python weld_test/weld_corner.py --tool N --execute --method spline   # 5. compare
```

- `N` = the torch tool number that was **active when p1..p4 were recorded**; also pass
  `--user` if a workpiece frame other than 0 was active. The script refuses to move if the
  robot's active tool or workpiece differ.
- The robot must be in **AUTOMATIC** mode (set it on the pendant).
- Each run: MoveJ to 30 mm above p1 (along the torch axis) → MoveL down to p1 → seam →
  retract 30 mm. It asks for `y` before moving.

### Stopping it (e.g. it is about to hit the part)

While the robot moves, press **ENTER** (or **Ctrl+C**) in the terminal. `StopMotion` is
sent at once, on a fresh connection to the controller, so it still gets through while
the script is waiting inside a robot command. The script also stops by itself if the
robot reports a fault (its own collision detection, the e-stop, …).

After a stop, the script sends **no** more motion: no retract. Jog the robot clear on the
pendant. If the terminal ever says `StopMotion FAILED`, press the e-stop. The terminal stop
is a convenience and doesn't replace the e-stop. For automatic stops on contact, set the
robot's collision detection level on the pendant.
- `--vel` is the seam speed in % (default 5). Raise it step by step. Compare methods at the
  same `--vel` and `--corner_mm`.

Useful options: `--corner_mm`, `--step_mm`, `--blend_mm`, `--ctrl_blend_mm`, `--blend_mode`,
`--approach_mm`, `--approach_vel`, `--spline_avg_time`, `--max_joint_step`, `--points`.
`--help` lists them all.

## Not verified yet (needs the real robot)

- How smoothly this controller firmware blends many 1 mm MoveL moves, compared with the spline.
- Exactly how `NewSplinePoint`'s `ovl` and `NewSplineStart`'s `averageTime` set the speed.
  Start with a low `--vel`.
- The `--fir` limit ranges.

The geometry and the SDK calls were checked offline. The SDK calls were tested against a
fake robot that uses the exact method signatures from `fairino_driver/Robot.py`.
