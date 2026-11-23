# suction_pick: multi-cup suction picking

Picks objects with a multi-cup vacuum gripper on the Fairino cobot, using the wrist Gemini 2.
The planner follows **Jiang et al., "Multiple-object Grasping Using a Multiple-suction-cup
Vacuum Gripper in Cluttered Scenes", arXiv:2304.10693**. It finds poses where several cups
land on graspable surfaces at once. The surfaces can belong to one large object, or to several
small objects that are picked together.

```
suction_pick/
├── __main__.py / cli.py   python -m suction_pick demo | plan | pick
├── config.py              gripper geometry and every tuning parameter
├── affordance.py          where ONE cup can seal (stands in for the paper's SG-U-Net++)
├── planner.py             the paper's multi-cup planner (voxels, kernels, ranking) + collision check
├── geometry.py            depth -> points, frames
├── camera.py              Gemini 2 capture (reuses cube_pick's V4L2 camera code)
├── robot.py               Fairino pick sequence (reuses cube_pick's camera->robot calibration)
├── visualize.py           overlay of graspable regions and planned cups
├── synthetic.py           test scene with known answers
└── tests/test_planner.py  offline tests
```

## How it works

| Step | Paper | Here |
|---|---|---|
| Affordance map | SG-U-Net++ (trained network; code and weights not published) | **Geometric:** the patch under a cup must be complete (no holes or edges), flat (plane-fit RMS ≤ 2 mm or 0.3 % of the distance), at most 45° tilted, and above the table (the largest horizontal plane) |
| Voxel grid | l = 5 mm, voxel occupied if it holds > 10 points | l = 5 mm, ≥ 3 points (the camera is farther away, so there are fewer points per voxel) |
| Orientations | top 10 % of normal bins (Δα = 5°) × roll | same, **plus** each region's own dominant normal (otherwise a lone tilted object is missed) |
| Cup placement | encoded kernel (cup i = 10⁻ⁱ), 3D convolution, decode digit i | the same, computed as one grid shift per cup (exact; equivalent, because the kernel has one voxel per cup) |
| Normal check | ε₁ = 11.5°, ε₂ = 10 mm | same |
| Ranking | most objects first, then J = J_center + J_var + J_orient | same order. J_center and Poly_orient are undefined in the paper; see `planner.py` for the definitions used here |
| Approach axis | the sampled 5° bin direction | **added:** after the checks, aligned with the measured surface normal under the cups, then rechecked. A parcel tilted in pitch and roll is approached within ~0.1° instead of up to ~2.5° |
| Cup contact | voxel-level (5 mm) | **added:** every active cup must touch its surface (gap ≤ `max_seal_gap` 4 mm) |
| Collision | none | **added:** reject a grasp if scene points reach into the gripper's footprint above the cup plane |
| Fallback | single cup at the best point | same, only with `--independent_cups` |

**Shared vacuum line (the default):** every cup must be on a graspable surface, otherwise
the free cup leaks and the grip fails. The object, or a group of objects at the same height,
must therefore be large enough for the whole cup layout. Only use `--independent_cups` if
each cup has its own valve.

## Tilted parcels (pitch and roll)

The gripper is oriented by the parcel's top surface, not by the bin floor:

- **Approach axis = surface normal.** It is measured from depth under the cups, so pitch and roll are
  both followed. Tool Z points into the surface: `R_tool = R_gripper · diag(1, −1, −1)`.
- **Rotation about that axis** (where the two cups point) is chosen by the planner: it samples it in
  5° steps and prefers the cups along the parcel's long side.
- **Motion:** pre-grasp is `--approach_mm` back **along the normal**, so the cups come straight
  down onto the slope. The grasp presses `--press_mm` along the normal, and the lift goes
  **straight up** (base Z).
- **Limit:** `--max_tilt` (45°). Steeper surfaces are not offered. The robot must also reach the
  tilted pose; the dry run checks IK for every pose and falls back to the next grasp.
- With the robot, tilt is measured relative to the **robot base** Z, so a bin that is not level is
  handled automatically: only the parcel's own surface matters.

### Is an object tilted?

Every `plan`/`pick` run prints the top-surface tilt of each detected object, relative to the
**table / bin floor** (its fitted plane), so a camera or bin that is not level doesn't count:

```
[OBJECTS] top-surface tilt vs table plane (pitch along X, roll along Y; ...; level = within 5 deg):
  region 1: level  tilt   0.3 deg (pitch -0.2, roll +0.2) at [-34.8, 59.9, -520.0] mm
  region 4: TILTED tilt  20.0 deg (pitch +20.0, roll +0.0) at [112.4, -79.4, -539.0] mm
```

Positive pitch means the surface rises towards +X; positive roll means it rises towards +Y.
X/Y are the robot base axes in `pick`, or image right/up in `plan`. The same values are written
to `grasps.json` → `"objects"` and onto `overlay.png` (red `TILT ...` / white `level ...`).
Change the level threshold with `--level_tol` (default 5°). Only objects with a graspable
(flat, sealable) top are listed.

## One parcel per pick, then home

- By default a grasp must have **all cups on the same parcel** (`max_objects = 1`). Parcels too
  small for the whole cup layout are skipped. `--multi_object` brings back the paper's mode,
  where one grasp can take several parcels side by side.
- Each `pick --execute` run picks **one** parcel, places it (with `--place`) or holds it, then
  **returns to the home pose** (`HOME_POSE` in `cube_pick/robot/home_pos.py`) at `--home_vel`
  (20 %). It also goes home after a failed vacuum check. `--no_home` skips this.
  Before measuring, the robot also **goes home first** (asks `y`; skipped if it is already there),
  so the scene is always seen from the same pose. `--no_start_home` measures from where it is.
- Run the command again for the next parcel: it re-measures the scene each time.

## Simulation in RViz2 (no robot needed)

```bash
python run.py sim                                   # newest saved pick_* capture, opens RViz2
python run.py sim --capture output/pick_YYYYMMDD_HHMMSS --bin_height 150 --arm_margin 50
```

Runs the **real planning code** on a saved capture (the grasp planner, the virtual walls from
`bin.json`, `plan_route` for the whole arm) and animates the pick in a loop: home → (via) → hover →
straight down → suction → lift with the parcel → home → release. The controller IK is replaced by
a numerical IK on the fitted FR10 model (`arm.py`, matches the controller to 0.0000 mm).

RViz shows (fixed frame `base_link`): the capture as a coloured point cloud, the board, the
virtual walls (orange) and the stand (grey), the FR10 as its collision capsules coloured by distance
to the cubicle (green > 50 mm, yellow 10–50, red < 10), the suction tool / cups / camera, the
parcel being carried, and the current phase with the closest distance. Topics:
`/suction_sim/markers`, `/suction_sim/cloud` (`--no_rviz` publishes only).

## Run

```bash
source /home/shashank/Projects/spark_src/.venv/bin/activate
cd /home/shashank/Projects/FoundationPose

python -m suction_pick demo --show            # synthetic scene, prints PASS/FAIL
python -m suction_pick plan --show            # live camera, plan only (no robot)
python -m suction_pick plan --replay suction_pick/output/plan_XXXX   # re-plan a saved capture

# robot: arm at a look pose above the objects (camera = tool 5, suction gripper = tool 1)
python -m suction_pick pick                                      # dry run: plan + IK check, no motion
python -m suction_pick pick --vacuum_do 2 --execute              # pick ONE parcel, hold it, go home
python -m suction_pick pick --vacuum_do 2 --execute --place 300 0 200 180 0 0
python -m unittest discover -s suction_pick/tests -t . -v        # offline tests
```

From **inside** the `suction_pick` folder, use `python run.py` instead of
`python -m suction_pick`, e.g. `python run.py pick --vacuum_do 2 --execute`.

Every run writes `suction_pick/output/<run>/`: `overlay.png` (graspable regions + top
grasps; a crossed circle means that cup is off), `affordance.png`, and `grasps.json`
(TCP, cup positions, normal, tool rotation, scores). Live runs also save `color.png`,
`depth_mm.npy`, `K.npy` and `T_plan_cam.npy`, so they can be replayed. Only the **newest 5 runs** are kept: older
run folders are deleted automatically after each run (`KEEP_RUNS` in `cli.py`).

The gripper defaults to **2 cups, 70 mm apart, Ø30 mm**, along the tool X axis. Change this with
`--cup_spacing`, `--cup_diameter`, or `--cups "x,y;x,y;..."` for other layouts.

Pick options: `--vacuum_do` takes 1 output (shared) or 1 per cup; `--tool_io` uses the flange
outputs (`SetToolDO`); `--vacuum_di` checks a vacuum sensor before lifting; `--approach_mm 60`,
`--press_mm 3`, `--lift_mm 100`, `--vacuum_wait 0.5`.
Tuning: `--max_range` (default 1200 mm: depth farther from the camera is ignored, so the floor
and room behind the table never count; at home the table is ~0.73 m away, the floor ~1.65 m),
`--flatness`, `--min_height` (0 = the table counts too), `--max_tilt`, `--voxel`,
`--min_points`.

**Robot requirements:** automatic mode; tool numbers are `CAM_TOOL = 5` and `SUCTION_TOOL = 1`
in `config.py` (override with `--cam_tool` / `--suction_tool`). Any tool may be active on the
pendant. The suction tool's TCP must be set on the controller
at the **midpoint between the cups, in the cup-lip plane, Z pointing out of the gripper**.
Ctrl+C sends `StopMotion` and leaves the vacuum as it is, so a held object isn't dropped.

## Status

- Offline: the synthetic demo and all 5 tests pass (multi-object grasp, large box, 20°-tilted box,
  parcels tilted in pitch and roll up to 40° (gripper axis within 0.5°, checked through the Fairino
  rx/ry/rz pose), collision with a tall neighbour, small objects with shared vs. independent cups).
- Live camera: planned in ~3 s on a real capture (the teal block on the board), with both cups
  along its long axis.
- Robot sequence: checked against a fake robot only. **Not yet run on the arm.** Do a dry run,
  then `--execute` without `--place`, with a hand on the e-stop.
- Not covered (same as the paper): touching objects at the same height merge into one region;
  the geometric affordance can't tell porous or very light objects apart.
