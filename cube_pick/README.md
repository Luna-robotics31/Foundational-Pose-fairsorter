# Cube pick with FoundationPose + Fairino cobot

> **Status (2026-11):** this is the older FoundationPose cube pick for a **two-finger gripper** on
> tool 1. Tool 1 is now the suction cup, so this pick does not match the current hardware. For picking
> parcels with the suction gripper see the main `readme.md` and `suction_pick/`. Install: Python 3.10,
> `requirements-pick.txt` plus the optional cube-pick packages listed at its end.

An Orbbec Gemini 2 camera on the robot's gripper finds a white recessed cube. FoundationPose
estimates the cube's 6D pose, and the Fairino cobot picks it up with a two-finger gripper.

All cube-pick code lives in this `cube_pick/` package; the surrounding FoundationPose
checkout (`readme.md` one level up) is used unchanged as a library.

---

## 1. How it works

```
Gemini 2 (RGB + depth)
   │
   ▼
vision/live_tracker.py  YOLO finds the cube → depth mask → FoundationPose 6D pose
   │                    writes debug/live/latest_pose.json (camera frame, metres)
   ▼
robot/pick_cube.py
   │                    camera pose = robot flange × camera tool (tool 5)
   │                    cube in robot base frame → grasp plan (hover → descend → close → lift)
   ▼
Fairino cobot (192.168.58.2), gripper tool 1
```

`start_pick.py` (`python -m cube_pick`) runs both in order, so you only need one command.

---

## 2. Folder layout

```
FoundationPose/                     upstream FoundationPose (estimater.py, Utils.py, weights/, mycpp/)
├── cube_pick/
│   ├── __main__.py                 `python -m cube_pick` → start_pick.main()
│   ├── start_pick.py               one-command launcher (tracker + pick)
│   ├── config.py                   all shared paths + robot IP
│   ├── vision/
│   │   ├── live_tracker.py         Orbbec camera + YOLO + FoundationPose; writes the pose file
│   │   └── pose_coordinates.py     pose → JSON record helpers
│   ├── robot/
│   │   ├── pick_cube.py            pose → robot base frame → grasp; moves the robot
│   │   ├── home_pos.py             optional home pose (--go_home)
│   │   └── fairino_driver/         Fairino Python SDK (Robot)
│   ├── assets/
│   │   ├── cube_recessed.obj       cube model (40 × 40 × 45 mm, Ø33 × 20 mm pocket), metres
│   │   ├── cube_recessed.blend     Blender source of the model (+ .png renders)
│   │   └── kaushal.pt              YOLO detector (class `white_cube`)
│   └── tools/
│       └── create_recessed_cube_blender.py   rebuilds the model into assets/
└── debug/live/                     runtime output (git-ignored)
    ├── latest_pose.json            latest cube pose (rewritten every frame)
    └── run_live.log                tracker log when started by start_pick
```

---

## 3. Requirements

- **Python environment:** `/home/shashank/Projects/spark_src/.venv` (PyTorch with CUDA,
  pyorbbecsdk, ultralytics, trimesh). It stays in spark_src; this folder just uses it.
- **NVIDIA GPU with CUDA** (FoundationPose won't run on CPU).
- **Gemini 2** on a **USB 3** port.
- **Fairino cobot** reachable over Ethernet:
  - laptop connection `farino_cobot` = static IP `192.168.58.24/24` on `enp3s0`
  - robot = `192.168.58.2`
- **Robot tool coordinates on the controller:**
  - **tool 5** = camera TCP (centre of the RGB lens). Its X/Y/Z are used; its rotation is
    replaced by the measured value `CAM_RPY_DEFAULT = (22.96, 1.09, 45.3)` in the pick script.
  - **tool 1** = gripper TCP (between the fingertips).
  - The tool that's *active* on the pendant doesn't matter.

Activate the environment in every new terminal:

```bash
source /home/shashank/Projects/spark_src/.venv/bin/activate
```

---

## 4. Quick start

1. Check that the robot answers: `ping 192.168.58.2`
2. Move the robot to the **look pose**: the camera looking down at the table, cube in view.
   The pick always measures from wherever the arm is when it starts.
3. Put the cube on the table.
4. Run, in this order:

```bash
cd /home/shashank/Projects/FoundationPose

python -m cube_pick                           # 1. dry run: prints the plan, robot does NOT move
python -m cube_pick --execute --hover_only    # 2. moves above the cube and stops (check by eye)
python -m cube_pick --execute                 # 3. full pick (type 'y' to confirm)
```

Keep a hand on the **e-stop** for the first runs. The script switches the robot to
**automatic mode** when it moves, so switch back to manual on the pendant before jogging.

### What the pick does

1. Waits for the arm to stop and reads 10 fresh cube poses (averaged; refuses if they disagree by > 3 mm).
2. Converts them to robot coordinates using the flange pose + camera tool 5.
3. Turns the gripper to the cube's angle (fingers on two faces; the cube repeats every 90°, so it picks the smallest wrist turn).
4. Opens the gripper → hover (≥ 250 mm, or 80 mm above the grasp) → slow descent to the cube centre → close → lift.

---

## 5. Useful options

Any option `start_pick.py` doesn't know is passed on to the pick script.

| Option | Effect |
|---|---|
| `--execute` | Actually move the robot (without it: dry run) |
| `--hover_only` | With `--execute`: stop above the cube |
| `--check` | No motion: keep printing the cube position, angle, table height (Ctrl+C to stop) |
| `--headless` | No camera preview window |
| `--height_source pose\|measured` | Grasp height from the FoundationPose cube centre (default) or from detected table + measured height |
| `--table_z Z` / `--table_z_flange Z` | Override the detected table height (gripper-tip Z / flange Z) |
| `--check_tilt` | Refuse cubes tilted > 15° (default: assume flat on the table) |
| `--go_home` | With `--execute`: go to `home_pos` first |
| `--keep_tracking` | Leave the tracker running after the pick |
| `--track_timeout S` | Seconds to wait for the first cube pose (default 90) |

Settings at the top of `robot/pick_cube.py`:

| Setting | Meaning |
|---|---|
| `GRASP_Z_OFFSET_MM` | Grip higher (+) / lower (−) than the cube centre |
| `GRIPPER_YAW_OFFSET_DEG` | Set to 45 if the fingers land on the cube's corners instead of its faces |
| `MIN_HOVER_Z_MM` | Lowest hover / lift height (250) |
| `APPROACH_MM`, `LIFT_MM` | Hover distance above the grasp / lift distance |
| `VEL_TRAVEL`, `VEL_DESCEND`, `VEL_LIFT` | Speeds (%) |
| `CAM_RPY_DEFAULT` | Measured camera rotation relative to the flange |

---

## 6. Running the parts separately

```bash
# Terminal 1: tracker with preview window (Q quits, R re-acquires)
cd /home/shashank/Projects/FoundationPose
python -m cube_pick.vision.live_tracker

# Terminal 2: pick script
cd /home/shashank/Projects/FoundationPose
python -m cube_pick.robot.pick_cube --cam_tool 5 --gripper_tool 1            # dry run
python -m cube_pick.robot.pick_cube --cam_tool 5 --gripper_tool 1 --execute
```

---

## 7. Changing the cube

1. Edit the dimensions at the top of `tools/create_recessed_cube_blender.py`
   (`WIDTH_MM`, `HEIGHT_MM`, `HOLE_DIAMETER_MM`, `RECESS_DEPTH_MM`, `BOTTOM_POCKET`).
2. Rebuild the mesh (written to `assets/`):
   ```bash
   cd /home/shashank/Projects/FoundationPose
   ~/Setups/blender-5.2.1-linux-x64/blender -b --factory-startup --python cube_pick/tools/create_recessed_cube_blender.py
   ```
3. Restart the tracker. It logs `Mesh dimensions (mm): [...]` at startup.

The mesh **must match the real part**. A wrong size makes the pose come out tilted and too low.

---

## 8. Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `[START] These processes would block ...` | An old tracker / pick / `main.py` is running or **paused (Ctrl+Z)**. Stop it (`kill -9 <PID>` for paused ones). Always quit with Ctrl+C or Q, never Ctrl+Z. |
| `GetToolCoordWithID(...) failed with error code -4` | The laptop isn't connected to the robot. Check the Ethernet cable, robot power and `ping 192.168.58.2`. |
| Script hangs right after `RobotUdpDataRecvThread 已启动` | Another program holds the robot's status port (often a paused `main.py`). Kill it. |
| `No valid cube pose within 90 s` / `Searching for cube with valid depth` | Cube not visible from the current arm pose, or not detected. Move to the look pose, check the preview window, press R. Detection of cubes on raised stands can be intermittent. |
| `No v4l device found` / `/dev/video*` missing | The camera was left in a bad state by an earlier program. The tracker reboots the camera automatically; if that fails, unplug and replug it. |
| `Grasp z ... is below the safety floor` | The height looks wrong: check the mesh matches the cube, or try `--height_source pose`. |
| `Robot cannot reach the ... pose(s) (no IK solution)` | The cube is outside the arm's reach for a top-down grasp. Move the cube closer. |
| `Pose samples disagree by ... mm` | The cube or the arm moved while measuring. Keep both still. |
| Gripper ends up off the cube | Camera tool position/rotation is off. Run `--check`, compare with the real position, and recalibrate tool 5 if needed. |

### Safety

- Always do a dry run and a `--hover_only` run after changing anything.
- The table (or the cube's own underside) is used as a floor: the gripper won't be sent below it.
- Don't put your hands near the gripper while `--execute` is running.
