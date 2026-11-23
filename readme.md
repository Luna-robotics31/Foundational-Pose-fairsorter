# Suction picking with a Fairino FR10 and an Orbbec Gemini 2

This repository picks parcels **one at a time** from a table or bin with a **two-cup suction gripper**
on a Fairino FR10 cobot. A Gemini 2 depth camera on the robot's wrist looks at the table, the program
finds a flat spot where both cups can seal, and the robot picks the parcel and brings it to its home
pose. Before anything moves, it checks that the whole arm stays clear of the table edges and of
**virtual bin walls**, and you can watch the planned pick in an **RViz simulation**.

If you are new here, follow this file from top to bottom. Every command below is copy-paste ready.

```
 Install (once) ─► Hardware check (once) ─► Camera calibration (once per camera mounting)
      ─► For every table:  set home ─► measure the bin ─► dry run ─► simulate in RViz ─► execute
```

---

## Contents

1. [What is in this repository](#1-what-is-in-this-repository)
2. [Hardware and network](#2-hardware-and-network)
3. [Install](#3-install-once)
4. [First hardware check](#4-first-hardware-check-once)
5. [Camera calibration](#5-camera-calibration-once-per-camera-mounting)
6. [Set up a table: home pose and bin](#6-set-up-a-table-home-pose-and-bin-detect-the-bin-first)
7. [Dry run](#7-dry-run-nothing-moves)
8. [Simulate in RViz](#8-simulate-in-rviz-no-robot-needed)
9. [Execute the pick](#9-execute-the-pick)
10. [Everyday checklist](#10-everyday-checklist)
11. [Files and settings](#11-files-and-settings)
12. [Useful options](#12-useful-options)
13. [Troubleshooting](#13-troubleshooting)
14. [Other folders (cube pick, welding test, segmentation lab)](#14-other-folders)
15. [Credits and licence](#15-credits-and-licence)

---

## 1. What is in this repository

| Folder / file | What it is |
|---|---|
| **`suction_pick/`** | **The suction pick.** Everything in this README. Run it from inside this folder with `python run.py …` |
| `cube_pick/` | Shared robot and camera code used by the suction pick (robot connection, tool data, home pose, camera driver, Fairino SDK in `cube_pick/robot/fairino_driver/`), plus the older FoundationPose cube pick (section 14) |
| `requirements-pick.txt` | Python packages for the suction pick |
| `requirements.txt` | FoundationPose's own packages (only for the cube pick) |
| `wield_test/`, `parcel_segmentation_lab/` | Separate experiments (section 14) |
| `estimater.py`, `learning/`, `mycpp/`, `weights/`, … | The FoundationPose 6D pose model (NVIDIA); only the cube pick uses it |

Inside `suction_pick/`:

| File | Purpose |
|---|---|
| `run.py` | Entry point: `python run.py demo \| plan \| bin \| calibrate \| sim \| pick` |
| `affordance.py`, `planner.py` | Finds where the cups can seal and plans the grasp (after Jiang et al., arXiv:2304.10693) |
| `workcell.py` | Measures the table and builds the virtual bin walls |
| `arm.py`, `fence.py` | Whole-arm collision model of the FR10 and the live wall monitor |
| `robot.py` | Pick sequence on the robot (moves, suction outputs, home) |
| `calibrate.py` | Camera calibration |
| `sim_rviz.py` | RViz2 simulation |
| `suction_activate.py` | Switches one digital output on and off (to test the suction) |
| `bin.json`, `camera_calibration.json` | Saved bin and camera calibration (**specific to this robot and table**: redo them for yours) |
| `output/` | One folder per run: images, depth, `overlay.png`, `grasps.json` (newest 5 kept) |

---

## 2. Hardware and network

| Item | Setting used by the code |
|---|---|
| Robot | Fairino **FR10**, controller software v3.8.5, IP **`192.168.58.2`** (`ROBOT_IP` in `cube_pick/config.py`) |
| Laptop | Ubuntu 22.04, Ethernet to the robot with a static address in `192.168.58.0/24`, e.g. `192.168.58.24` |
| Camera | Orbbec **Gemini 2** on the wrist, on a **USB 3** port |
| Gripper | 2 suction cups, Ø30 mm, 70 mm apart, one shared vacuum line |
| Suction outputs | Control-box **DO0** = suction valve, **DO1** = release (blow-off) |
| Tool 1 (pendant) | Suction TCP: midpoint between the cups, in the cup-lip plane: `[2.333, 0.564, 127.237, 0, 0, 0]` |
| Tool 5 (pendant) | Camera TCP (centre of the RGB lens): `[5.933, -77.174, 30.342, -15.248, 10.272, -144.596]` |

The tool values above are stored in `TOOL_COORDS` in `cube_pick/config.py`. This controller version
cannot report tool data by number, so **the file must match the pendant**. If you re-teach a tool,
update the file. At start-up the pick compares the file with the controller and stops if they differ.

**Network:** set the laptop's Ethernet port to a static address. Find the port name with `ip link`
(here `enp3s0`):

```bash
sudo nmcli con add type ethernet ifname enp3s0 con-name fairino \
     ipv4.method manual ipv4.addresses 192.168.58.24/24
sudo nmcli con up fairino
ping -c 3 192.168.58.2        # must answer
```

---

## 3. Install (once)

### 3.1 Python environment

Python **3.10** is required: the bundled Fairino SDK is built for CPython 3.10, and ROS 2 Humble uses
3.10 too. Ubuntu 22.04 ships it.

```bash
sudo apt install python3.10-venv
cd /path/to/FoundationPose                   # this repository
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pick.txt
pip install --no-deps pyorbbecsdk2==2.1.2    # camera driver, installed on its own (see below)
```

Install the camera driver **with `--no-deps`**. Its package metadata asks for `numpy<2` and some
libraries for its example scripts, which would conflict with OpenCV 5. The driver itself only needs
numpy and works with the numpy 2.2.6 installed above. Because of this, `pip check` lists warnings
about `pyorbbecsdk2` (numpy, pygame, pynput): these are expected and harmless.

> **On the lab laptop** the environment already exists:
> `source /home/shashank/Projects/spark_src/.venv/bin/activate`

Activate the environment in **every new terminal** before using the commands below.

### 3.2 Camera USB permissions

The driver includes a script that installs the USB rule for Orbbec cameras:

```bash
sudo bash "$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('pyorbbecsdk').origin))")/shared/install_udev_rules.sh"
```

Then unplug and replug the camera. Check that it is seen with `lsusb | grep -i orbbec`.

### 3.3 ROS 2 Humble (only for the RViz simulation)

Check if it is already installed: `ls /opt/ros/humble`. If not, follow the official instructions
(<https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>) and install the desktop
version, which includes RViz2:

```bash
sudo apt install ros-humble-desktop
```

You do not need to `source` ROS yourself. `python run.py sim` restarts itself inside the ROS
environment.

### 3.4 Check the installation (no robot, no camera)

```bash
cd suction_pick
python run.py demo                                  # synthetic scene
cd .. && python -m unittest discover -s suction_pick/tests -t . && cd suction_pick
```

`demo` must print five `PASS:` lines, and the tests must end with `OK`.

---

## 4. First hardware check (once)

All commands from here on are run **from the `suction_pick` folder** with the environment active.

| Check | Command | Expected |
|---|---|---|
| Robot answers | `ping -c 3 192.168.58.2` | replies |
| Robot readout (no motion) | `python -m cube_pick.robot.home_pos`; if it asks to move, answer **`n`** | prints `[HOME] flange now [...]` and the home pose |
| Camera (no robot) | `python run.py plan --show` | a window with the colour image, graspable areas and planned cups (Q closes it) |
| Release output | `python suction_activate.py` | `control box DO1 ... now: 1`, then off after 2 s |
| Suction output | `python suction_activate.py --do 0` | the suction runs for 2 s |

`python -m cube_pick.robot.home_pos` works from the repository root. From `suction_pick/`, run it as
`(cd .. && python -m cube_pick.robot.home_pos)`.

---

## 5. Camera calibration (once per camera mounting)

The program has to know exactly where the camera sits on the arm. Redo this whenever the camera or the
gripper is taken off, moved or re-mounted. It is **not** needed when you change tables.

1. Put an object on the table: something flat-topped, at least 10 cm across and about 5 cm tall (the
   white foam disc works). It must not move during calibration.
2. Move the robot to its home pose (section 6.2), with the object **fully in view near the image
   centre**. Clear the area around the robot.
3. Run:
   ```bash
   python run.py calibrate --tilt 10
   ```
   Type `y`. The robot visits 9 poses, tilted 10° around the object at 10 % speed, and comes back.
   If it says a pose is unreachable, use a smaller `--tilt`.
4. Read the result:
   ```
   [CALIB] ROTATION: board normals agree within 0.10 deg ...     <- under 0.5 deg is good
   [CALIB] POSITION: camera xyz [8.9, -58.3, 27.5] mm ...
   [CALIB] CHECK (leave-one-out ...): ... mean 4.4 mm, max 7.5 mm  <- under 3 mm is very good; above 8 mm it is not saved
   [CALIB] Saved camera_calibration.json ...
   ```

The result is saved in `suction_pick/camera_calibration.json`, and every pick uses it.

---

## 6. Set up a table: home pose and bin (detect the bin first)

Do this for **every new table**, and whenever a table is moved. The robot must know where the table
is **before** you execute a pick.

### 6.1 Point the camera at the table

Jog the robot on the pendant until the camera looks **roughly straight down** at the table from about
**600–800 mm above the table top**, with **all four table edges** and the parcels in view.

### 6.2 Save that pose as home

```bash
(cd .. && python -m cube_pick.robot.home_pos)     # if it asks to move, answer n
```

Copy the six numbers printed after `[HOME] flange now` into `HOME_POSE` in
`cube_pick/robot/home_pos.py`. Use the **flange** values printed here, not the TCP pose shown on the
pendant (that one depends on which tool is active). Every pick starts at, measures from and returns
to this pose.

### 6.3 Measure the table: this creates the bin

```bash
cp bin.json bin_previous.json      # optional: keep the old bin
python run.py bin
```

The program takes the **nearest large flat surface** as the table, not the floor around it, and
saves `bin.json`. Example output:

```
[BIN] board 563 x 453 mm, flatness 2.1 mm, surface at z -353 mm
[BIN] floor below it at z -642 mm (table 288 mm high): the arm check treats the table as solid down to the floor
[BIN] Saved bin.json. Every pick now adds walls --bin_height mm high (default 100).
```

Check that:
- **the size matches the real table** (measure it with a tape). If it is much larger, the camera saw
  the floor: move the camera closer to the table and measure again;
- no side is reported **outside the camera view**. If one is, the command does not save. Move the
  camera higher or further back, or use `python run.py bin --allow_partial`, which puts that wall
  where the view ends.

### What the bin does from now on

- Only things **on top of the table and inside its outline** are treated as parcels.
- **Virtual walls** stand along the table edges, `--bin_height` mm high (default **100 mm**). Neither
  the gripper nor any part of the arm may touch them, and the robot approaches from above them.
- The table body below the surface (down to the floor) is treated as solid for the arm.

---

## 7. Dry run (nothing moves)

```bash
python run.py pick
```

The robot does not move. The program takes a picture (measuring from the current pose and saying it
would first go home), plans the grasp and the whole route, and saves everything in
`output/pick_<date>_<time>/`. Check these lines:

```
[CHECK] flange·tool vs controller TCP: 0.00 mm, 0.00°           tool data matches the pendant
[BIN] virtual walls 100 mm high around the board 563 x 453 mm   the bin from section 6 is used
[OBJECTS] region 1: level tilt 0.3 deg ... at [580.1, -729.8, -303.2] mm    a parcel was found on the table
[ARM] whole-arm distance to the cubicle per move: home -> pre-grasp (joint move): 176 mm (wrist 3) | ...
                                                                 every move well above 30 mm
[PICK] grasp #1: 1 object(s), cups active [1, 1], tilt 1.4 deg   the chosen grasp
[DRY RUN] Nothing moved.
```

Open **`output/pick_<…>/overlay.png`**:
- **yellow line:** table edge; **orange line:** top of the virtual walls;
- **green and yellow circles with a white cross:** the planned cups and their centre. They must be on
  the parcel you expect;
- **cyan / magenta circles:** tool body and camera; **orange tint:** obstacles near the path;
  **red:** something would be hit (a grasp with red is never chosen).

---

## 8. Simulate in RViz (no robot needed)

The simulation replays the scene from your **latest dry run**, so do section 7 first.

```bash
python run.py sim
```

RViz2 opens by itself and loops the pick: home → hover → straight down → suction → lift **with the
parcel** → home → release. It uses the same grasp planner, bin walls and whole-arm check as the real
pick. You see:

- the camera scene as a coloured 3D point cloud;
- the table (orange), the virtual walls (translucent orange) and the table body (grey);
- the FR10 drawn as its collision model, each link coloured by its distance to the table and walls:
  **green** > 50 mm, **yellow** 10–50 mm, **red** < 10 mm;
- the suction tool, cups and camera, the parcel being carried, and a text with the current step and
  the closest distance.

Close RViz or press Ctrl+C to stop. Useful variants:

```bash
python run.py sim --bin_height 150                      # try taller walls
python run.py sim --arm_margin 60                       # stricter arm clearance
python run.py sim --capture output/pick_20261119_183952 # replay an older dry run
```

If no safe pick exists, the terminal says why (for example `rejected: arm would come 12 mm from the
cubicle`), and RViz shows only the scene.

---

## 9. Execute the pick

Keep a hand on the **emergency stop**. Make sure nobody is near the robot.

```bash
python run.py pick --execute
```

What it asks:
1. `Type 'r' to reset the fault`: only if the robot shows a fault. `r` clears it.
2. `Robot is … mm from home. Type 'y' to move home`: only if the robot is not already at home.
3. It takes the picture, plans and prints the grasp and the route (as in section 7). Then
   `Type 'y' to pick`.

What the robot does:
1. Switches the controller to automatic mode.
2. Joint move to the **hover point** above the parcel (above the wall tops). **DO0 on** (suction starts).
3. Straight down onto the parcel, presses 3 mm, waits 0.5 s.
4. Lifts straight up, and does a joint move **home with the parcel**.
5. At home: **DO0 off and DO1 on** at the same moment (release). **DO1 off** after 2 s.

During every move the **live wall monitor** prints how close the arm came, e.g.
`[FENCE] grasp: closest wall 107 mm (cups)`. If any part comes within 10 mm of a wall it stops the
robot and prints `[FENCE] STOPPED: …`.

**To stop:** press **Ctrl+C** (the robot stops; the suction stays as it is, so a held parcel is not
dropped) or the **emergency stop**. Afterwards, jog the robot clear on the pendant, then send it home
with `(cd .. && python -m cube_pick.robot.home_pos)`.

One run picks **one** parcel. Run the command again for the next one.

---

## 10. Everyday checklist

```bash
source /home/shashank/Projects/spark_src/.venv/bin/activate    # or your .venv
cd /path/to/FoundationPose/suction_pick
ping -c 1 192.168.58.2 && lsusb | grep -i orbbec                # robot and camera present
python run.py pick                                               # dry run: parcel found? route OK?
python run.py sim                                                # optional: watch it in RViz
python run.py pick --execute                                     # pick one parcel
```

New table or table moved → section 6 first. Camera or gripper re-mounted → section 5 first.

---

## 11. Files and settings

| What | Where |
|---|---|
| Robot IP, tool numbers, tool coordinates | `cube_pick/config.py` (`ROBOT_IP`, `CAM_TOOL = 5`, `SUCTION_TOOL = 1`, `TOOL_COORDS`) |
| Home pose (flange pose) | `cube_pick/robot/home_pos.py` (`HOME_POSE`) |
| Bin (table outline, height, floor) | `suction_pick/bin.json`, written by `python run.py bin` |
| Camera calibration | `suction_pick/camera_calibration.json`, written by `python run.py calibrate` |
| Gripper and planner defaults | `suction_pick/config.py` |
| Robot speeds (travel 20 %, descend 8 %, lift 15 %) | `VEL_TRAVEL`, `VEL_DESCEND`, `VEL_LIFT` in `cube_pick/robot/pick_cube.py` |
| Run results | `suction_pick/output/pick_<time>/` (newest 5 kept; `KEEP_RUNS` in `suction_pick/cli.py`) |

---

## 12. Useful options

For `python run.py pick` (most also work for `sim`):

| Option | Default | Meaning |
|---|---|---|
| `--execute` | off | Actually move the robot (without it: dry run) |
| `--bin_height MM` | 100 | Height of the virtual walls above the table |
| `--no_bin` | off | Ignore the bin (no walls; any flat surface counts as the table) |
| `--arm_margin MM` | 30 | Planned routes keep every arm link at least this far from the table and walls |
| `--fence_margin MM` | 10 | Live monitor stops the robot this close to a wall |
| `--clearance_mm MM` | 30 | Hover point this far above the highest obstacle near the path |
| `--vacuum_do N` | 0 | Suction output |
| `--release_do N` | 1 | Release output (`-1` = none) |
| `--release_time S` | 2 | Seconds the release output stays on |
| `--place X Y Z RX RY RZ` | none | Release at this suction-tip pose instead of at home |
| `--no_start_home` | off | Measure from where the robot is now, instead of going home first |
| `--no_home` | off | Do not go home after the pick |
| `--home_vel %` | 20 | Speed of the moves to home |
| `--body_radius MM`, `--camera_radius MM` | 45, 50 | Size of the tool body and camera for the collision check |
| `--max_range MM` | 1200 | Ignore depth farther than this (extended automatically to reach the bin) |
| `--show` | off | Show the result image in a window |

`python run.py <command> --help` lists everything.

---

## 13. Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `No status data from the robot at 192.168.58.2` | Laptop not connected to the robot: check the Ethernet cable and `ping 192.168.58.2` (section 2). |
| `No Orbbec camera found: check the USB 3 cable` | Camera unplugged or USB rule missing: `lsusb \| grep -i orbbec`; replug; section 3.2. |
| `uvc_open failed ... Return Code: -6` | Another program is using the camera (a run still open in another terminal, the cube-pick tracker, the parcel segmentation lab, Orbbec Viewer). Close it. |
| `Too few depth points on the board` / `No large horizontal surface in view` | The camera does not see the table (too far, or pointing elsewhere). Move it 600–800 mm above the table. |
| `bin` reports a size much larger than the table | It measured the floor: the table is too small in the image. Move the camera closer to the table. |
| `[SKIPPED] A parcel is cut off at the image ... edge` | The parcel is not fully in view from home. Move the home pose (section 6) in the direction printed. |
| `No grasp found` / `no graspable surface found` | No flat spot fits both cups (the parcel needs about 10 cm across), or every route is blocked by the walls. Check `overlay.png`. |
| `[ARM] ... rejected: arm would come ... mm from the cubicle` | The arm would get too close to the table or walls on every route. Move the parcel away from the edge, lower `--bin_height`, or change the home pose. |
| `The table differs by ... deg from the board measured during calibration` | The camera calibration does not match, or the table is tilted: redo section 5, or use `--max_table_tilt`. |
| `Tool 1 (or 5) on the controller is [...] but TOOL_COORDS says [...]` | A tool was re-taught on the pendant. Update `TOOL_COORDS` in `cube_pick/config.py`. |
| Pendant error **1/33** "joint configuration of the next command changes", then MoveL error **185** | An old version used a straight move where the wrist had to flip. Clear the fault (`r` at the prompt, or Clear on the pendant). Current code uses joint moves for travel. |
| MoveL error **74** | Target point rejected. Usually tool data mismatch: check `TOOL_COORDS`. |
| The robot goes near the parcel but not to its centre | Camera calibration is off: redo section 5. |
| `[SIM] No saved scene yet` | Do a dry run first (`python run.py pick`), then `python run.py sim`. |
| RViz does not open | Check `suction_pick/output/rviz.log`. ROS 2 Humble must be installed (section 3.3). |

---

## 14. Other folders

### Cube pick with FoundationPose (`cube_pick/`, older demo)

The original demo of this repository: FoundationPose (a 6D pose network on the GPU) tracks a white
recessed cube, and the robot picks it with a **two-finger gripper** on tool 1. Tool 1 is now the
suction cup, so this pick does **not** match the current hardware. It is kept for reference and its
code is shared with the suction pick. It needs a CUDA GPU and the extra packages listed at the bottom
of `requirements-pick.txt`, then:

```bash
python -m cube_pick              # from the repository root; details in cube_pick/README.md
```

### `wield_test/`, `parcel_segmentation_lab/`

Separate experiments: a smooth-corner welding path test for the Fairino (`wield_test/README.md`) and a
parcel segmentation lab with its own README and requirements (`parcel_segmentation_lab/README.md`).
The suction pick does not use them.

---

## 15. Credits and licence

- Built on **FoundationPose** by NVIDIA (<https://github.com/NVlabs/FoundationPose>), under the NVIDIA
  Source Code License in `LICENSE`. The original FoundationPose README is in git history:
  `git show a1b694b:readme.md`.
- The multi-cup grasp planner follows Jiang et al., *Multiple-object Grasping Using a
  Multiple-suction-cup Vacuum Gripper in Cluttered Scenes*, arXiv:2304.10693.
- The Fairino Python SDK in `cube_pick/robot/fairino_driver/` is Fairino's.
- Detailed design notes of the suction pick: `suction_pick/README.md`.
