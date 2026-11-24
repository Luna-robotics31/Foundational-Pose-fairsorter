"""Commands of the suction pick. Start them with run.py in the suction_pick folder:

    python run.py demo                       # synthetic scene, no camera/robot
    python run.py bin [--auto]                # choose the bin on the camera image -> data/bin.json
    python run.py plan [--show]               # live camera, plan only (camera frame)
    python run.py plan --replay DIR           # replay a saved capture
    python run.py pick                        # robot: plan in base frame, dry run
    python run.py pick --execute             # suction DO0, release DO1 (2 s pulse)
(camera = tool 5, suction gripper = tool 1: CAM_TOOL / SUCTION_TOOL in cube_pick/config.py)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from suction_pick.vision.affordance import compute_affordance, object_tilts
from suction_pick.settings.config import CAM_TOOL, SUCTION_TOOL, AffordanceParams, Gripper, PlannerParams
from suction_pick.vision.geometry import T_UP_FROM_CAMERA, depth_to_points, transform
from suction_pick.planning.planner import plan
from suction_pick.vision.visualize import affordance_image, overlay

from suction_pick.settings.paths import OUTPUT_DIR as OUT_ROOT  # suction_pick/output
KEEP_RUNS = 5
RUN_NAME = re.compile(r"^(demo|plan|pick)_(\d{8}_\d{6})$")


def prune_outputs(keep=KEEP_RUNS, protect=()):
    """Delete all but the newest `keep` run folders (demo_/plan_/pick_<timestamp>) in OUT_ROOT.

    Folders in `protect` (the run just written, a replayed capture) are never deleted.
    """
    if not OUT_ROOT.is_dir():
        return []
    runs = sorted((d for d in OUT_ROOT.iterdir() if d.is_dir() and RUN_NAME.match(d.name)),
                  key=lambda d: RUN_NAME.match(d.name).group(2), reverse=True)
    protect = {Path(x).resolve() for x in protect}
    removed = []
    for d in runs[keep:]:
        if d.resolve() in protect:
            continue
        shutil.rmtree(d)
        removed.append(d.name)
    if removed:
        print(f"[OUT] removed {len(removed)} older run folder(s); keeping the newest {keep}")
    return removed


def build_config(args):
    if args.cups:
        cup_xy = [[float(v) for v in c.split(",")] for c in args.cups.split(";")]
    else:
        cup_xy = [[-args.cup_spacing / 2, 0.0], [args.cup_spacing / 2, 0.0]]
    from suction_pick.settings.config import robot_ee_geometry
    gripper = Gripper(cup_xy=cup_xy, cup_diameter=args.cup_diameter,
                      independent_cups=args.independent_cups, body_radius=args.body_radius,
                      camera_radius=args.camera_radius, **robot_ee_geometry())
    aff = AffordanceParams(flatness=args.flatness, min_height=args.min_height, max_tilt=args.max_tilt,
                           max_range=args.max_range)
    planner = PlannerParams(voxel=args.voxel, min_points_per_voxel=args.min_points,
                            max_tilt=args.max_tilt, max_objects=99 if args.multi_object else 1)
    return gripper, aff, planner


def run_pipeline(bgr, depth, K, points, T_plan_cam, args, out_dir: Path):
    gripper, aff_params, plan_params = build_config(args)
    walls, bin_, table, region = None, None, None, None
    if args.frame_name == "robot base frame" and not args.no_bin:   # virtual bin walls (robot coordinates)
        from suction_pick.robot import workcell
        bin_ = workcell.load()
        if bin_ is None:
            print("[BIN] No bin.json: no virtual walls. Measure the board with:  python run.py bin")
        else:
            walls = workcell.wall_points(bin_, args.bin_height)
            C, nb = bin_["corners_base"], bin_["normal_base"]
            print(f"[BIN] virtual walls {args.bin_height:g} mm high around the board "
                  f"{bin_['size_mm'][0]:.0f} x {bin_['size_mm'][1]:.0f} mm at z {C[:, 2].mean():.0f} mm "
                  f"(measured {bin_['date']})")
            # The bin surface is the table: parcels = things on top of it, inside its outline.
            table = (nb, -float(nb @ C.mean(axis=0)))
            region = workcell.inside_mask(bin_, points)
            # Reach the whole bin even if it is further than --max_range (the floor around it is
            # excluded anyway: it is below the bin surface and outside the outline).
            T_cam_base = np.linalg.inv(T_plan_cam)
            far = float((C @ T_cam_base[:3, :3].T + T_cam_base[:3, 3])[:, 2].max())   # farthest bin corner
            if aff_params.max_range > 0 and far + 50 > aff_params.max_range:
                aff_params.max_range = far + 50
                print(f"[BIN] working range extended to {aff_params.max_range:.0f} mm to reach the whole bin")
    t0 = time.monotonic()
    surface = compute_affordance(points, depth, K[0, 0], gripper.cup_radius, aff_params, table=table, region=region)
    t1 = time.monotonic()
    grasps, info = plan(surface, gripper, plan_params, obstacles=walls)
    t2 = time.monotonic()
    print(f"[AFFORDANCE] {surface.stats} ({t1 - t0:.1f} s)")
    for cut in surface.stats.get("cut_off_at_image_edge", []):
        # image direction -> planning-frame direction (camera X = image right, Y = image down)
        axis = {"left": -T_plan_cam[:3, 0], "right": T_plan_cam[:3, 0],
                "top": -T_plan_cam[:3, 1], "bottom": T_plan_cam[:3, 1]}
        d = sum(axis[e] for e in cut["edges"])
        d = d[:2] / (np.linalg.norm(d[:2]) or 1.0)
        print(f"[SKIPPED] A parcel is cut off at the image {'/'.join(cut['edges'])} edge (pixel "
              f"{cut['pixel_centre']}): not picked. Move the camera / HOME_POSE towards "
              f"({d[0]:+.2f}, {d[1]:+.2f}) in the {args.frame_name} XY so it is fully in view.")
    tilts = object_tilts(surface, args.level_tol)
    axes = ("robot base X/Y" if args.frame_name == "robot base frame"
            else "X = image right, Y = image up")
    print(f"[OBJECTS] top-surface tilt vs {tilts[0]['reference'] if tilts else 'table'} "
          f"(pitch along X, roll along Y; {axes}; level = within {args.level_tol:g} deg):")
    for t in tilts:
        print(f"  region {t['region']}: {'TILTED' if t['tilted'] else 'level '} tilt {t['tilt_deg']:5.1f} deg "
              f"(pitch {t['pitch_deg']:+.1f}, roll {t['roll_deg']:+.1f}) at {t['centre_mm']} mm")
    print(f"[PLAN] {info} ({t2 - t1:.1f} s)")
    for i, g in enumerate(grasps):
        print(f"  #{i + 1}: {g.n_objects} object(s), cups {g.active.astype(int).tolist()} on regions "
              f"{g.cup_labels.tolist()}, TCP {np.round(g.tcp, 1).tolist()} mm, tilt {g.tilt_deg:.1f} deg, "
              f"score {g.score:.1f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    T_cam_plan = np.linalg.inv(T_plan_cam)
    img = overlay(bgr, surface, grasps, K, T_cam_plan, gripper.cup_radius, tilts,
                  gripper=gripper, params=plan_params)
    if bin_ is not None:
        from suction_pick.robot import workcell
        workcell.draw(img, bin_, args.bin_height, K, T_cam_plan)
    cv2.imwrite(str(out_dir / "overlay.png"), img)
    cv2.imwrite(str(out_dir / "affordance.png"), affordance_image(surface))
    (out_dir / "grasps.json").write_text(json.dumps({
        "frame": args.frame_name, "units": "mm", "gripper": asdict(gripper),
        "affordance_params": asdict(aff_params), "planner_params": asdict(plan_params),
        "affordance": surface.stats, "objects": tilts, "planner": info,
        "grasps": [g.as_dict() for g in grasps],
    }, indent=2, default=str))
    print(f"[OUT] {out_dir}")
    if getattr(args, "show", False):
        cv2.imshow("suction grasps (Q to close)", np.hstack([
            cv2.imread(str(out_dir / "overlay.png")), cv2.imread(str(out_dir / "affordance.png"))]))
        while cv2.waitKey(50) & 0xFF not in (ord("q"), 27):
            if cv2.getWindowProperty("suction grasps (Q to close)", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
    return surface, grasps, info


def save_capture(out_dir, bgr, depth, K, T_plan_cam):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "color.png"), bgr)
    np.save(out_dir / "depth_mm.npy", depth)
    np.save(out_dir / "K.npy", K)
    np.save(out_dir / "T_plan_cam.npy", T_plan_cam)


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


# ── commands ────────────────────────────────────────────────────────────
def cmd_demo(args):
    from suction_pick.sim import synthetic as S
    args.frame_name = "synthetic camera-up frame"
    depth, _ = S.render()
    gray = cv2.cvtColor(cv2.normalize(-depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)
    points = transform(T_UP_FROM_CAMERA, depth_to_points(depth, S.K))
    out = OUT_ROOT / f"demo_{stamp()}"
    _, grasps, _ = run_pipeline(gray, depth, S.K, points, T_UP_FROM_CAMERA, args, out)
    prune_outputs(protect=[out])

    def on(g):
        return [S.box_under(c[:2]) for c, a in zip(g.cups, g.active) if a]
    checks = []
    best = grasps[0] if grasps else None
    if args.multi_object:
        checks.append(("best grasp takes small boxes A and B together",
                       best is not None and sorted(on(best)) == ["small_A", "small_B"]
                       and np.linalg.norm(best.tcp[:2] - [0, 60]) < 5 and abs(best.tcp[2] + 520) < 3))
    else:
        checks.append(("one parcel per grasp: no grasp spans two boxes",
                       bool(grasps) and all(len(set(on(g))) == 1 for g in grasps)))
        checks.append(("small boxes A, B (too small for both cups) are not picked",
                       not any(set(on(g)) & {"small_A", "small_B"} for g in grasps)))
    c = [g for g in grasps if on(g) == ["large_C", "large_C"]]
    checks.append(("both cups on the large box near its centre",
                   bool(c) and np.linalg.norm(c[0].tcp[:2] - [-110, -80]) < 10))
    d = [g for g in grasps if on(g) == ["tilted_D", "tilted_D"]]
    checks.append(("tilted box: approach follows its 20 deg tilt", bool(d) and abs(d[0].tilt_deg - 20) < 3))
    checks.append(("nothing on the 25 mm box (smaller than a cup)",
                   all("tiny_E" not in on(g) for g in grasps)))
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return 0 if all(ok for _, ok in checks) else 1


def cmd_plan(args):
    args.frame_name = "camera up frame (X right, Y up in image, Z towards camera)"
    if args.replay:
        d = Path(args.replay)
        if d.resolve().name.startswith("pick_"):  # captured by `pick`: saved in robot base coordinates
            args.frame_name = "robot base frame"
        bgr, depth, K = cv2.imread(str(d / "color.png")), np.load(d / "depth_mm.npy"), np.load(d / "K.npy")
        T_plan_cam = np.load(d / "T_plan_cam.npy")
        out = d / f"replan_{stamp()}"
    else:
        from suction_pick.vision.camera import capture
        bgr, depth, K = capture(args.frames)
        T_plan_cam = T_UP_FROM_CAMERA
        out = OUT_ROOT / f"plan_{stamp()}"
        save_capture(out, bgr, depth, K, T_plan_cam)
    points = transform(T_plan_cam, depth_to_points(depth, K))
    _, grasps, _ = run_pipeline(bgr, depth, K, points, T_plan_cam, args, out)
    prune_outputs(protect=[out, args.replay] if args.replay else [out])
    return 0 if grasps else 1


def cmd_bin(args):
    """Measure the board rectangle in robot coordinates and save it as the virtual bin."""
    from suction_pick.robot import workcell
    if args.replay:
        d = Path(args.replay)
        depth, K, T_base_cam = np.load(d / "depth_mm.npy"), np.load(d / "K.npy"), np.load(d / "T_plan_cam.npy")
        if not d.resolve().name.startswith("pick_"):
            raise SystemExit("--replay needs a pick_* capture (saved in robot coordinates).")
        bgr = cv2.imread(str(d / "color.png")) if (d / "color.png").exists() else None
    else:
        from cube_pick.config import CAM_TOOL, SUCTION_TOOL, TOOL_COORDS, connect_robot
        from cube_pick.robot.pick_cube import CAM_RPY_DEFAULT, Cell
        from suction_pick.robot.calibrate import load_cam_calib
        from suction_pick.vision.camera import capture
        robot = connect_robot()
        rpy, xyz, _ = load_cam_calib(CAM_RPY_DEFAULT, TOOL_COORDS[CAM_TOOL][:3])
        cell = Cell(robot, CAM_TOOL, SUCTION_TOOL, 0.0, rpy, cam_xyz=xyz)
        cell.wait_still()
        T_base_cam = cell.flange() @ cell.T_flange_optical
        bgr, depth, K = capture(args.frames)
    points = depth_to_points(depth, K)
    use_window = not args.auto and bgr is not None and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not args.auto and not use_window:
        print("[BIN] No display (or no colour image): automatic choice, no window.")
    if use_window:
        from suction_pick.app.bin_ui import choose
        b = choose(bgr, points, K, T_base_cam, args.max_range)
        if b is None:
            print("[BIN] Cancelled, nothing saved.")
            return 1
        print(f"[BIN] bin = {b['source']}")
    else:
        b = workcell.measure_board(points, K, T_base_cam, args.max_range)
    print(f"[BIN] board {b['size_mm'][0]:.0f} x {b['size_mm'][1]:.0f} mm, flatness {b['flatness_mm']:.1f} mm, "
          f"surface at z {b['corners_base'][:, 2].mean():.0f} mm")
    if b.get("floor_z") is not None:
        print(f"[BIN] floor below it at z {b['floor_z']:.0f} mm (table {b['corners_base'][:, 2].mean() - b['floor_z']:.0f} mm "
              "high): the arm check treats the table as solid down to the floor")
    for i, c in enumerate(b["corners_base"]):
        print(f"   corner {i}: {np.round(c, 1).tolist()} mm")
    if b["cut_sides"]:
        print(f"[BIN] side(s) {b['cut_sides']} (corner i -> i+1) run outside the camera view: there the wall "
              "is where the view ends, not at the real board edge.")
        if not args.allow_partial and not use_window:
            print("[BIN] NOT saved. Move the robot so the whole board is in view and run again, or add "
                  "--allow_partial to keep the visible part (safe: the robot stays inside what it saw).")
            return 1
    workcell.save(b)
    print(f"[BIN] Saved {workcell.BIN_FILE.name}. Every pick now adds walls --bin_height mm high (default 100).")
    return 0


def cmd_pick(args):
    from suction_pick.robot.pick import run
    args.frame_name = "robot base frame"
    if args.release_do >= 0 and args.release_do in args.vacuum_do:
        raise SystemExit(f"DO{args.release_do} is both the suction output (--vacuum_do) and the release output "
                         f"(--release_do). Suction is DO0 and release DO1 by default: just leave out --vacuum_do.")

    def plan_fn(bgr, depth, K, points, T_base_cam):
        out = OUT_ROOT / f"pick_{stamp()}"
        save_capture(out, bgr, depth, K, T_base_cam)
        surface, grasps, _ = run_pipeline(bgr, depth, K, points, T_base_cam, args, out)
        prune_outputs(protect=[out])
        if surface.table is not None:
            # A table that looks tilted in robot coordinates means the camera calibration is
            # wrong: every grasp would land in the wrong place.
            # Compared with the board normal measured during calibration (the real board may be
            # tilted a few degrees); without a calibration, with "level" (base Z).
            from suction_pick.robot.calibrate import CALIB_FILE
            ref, ref_name = np.array([0.0, 0.0, 1.0]), "level (robot base Z)"
            try:
                ref = np.array(json.loads(CALIB_FILE.read_text())["board_normal_base"], float)
                ref_name = "the board measured during calibration"
            except (OSError, KeyError, ValueError):
                pass
            n_t = surface.table[0]
            tilt = float(np.degrees(np.arccos(abs(n_t[2]))))
            dev = float(np.degrees(np.arccos(min(1.0, abs(n_t @ ref) / np.linalg.norm(ref)))))
            print(f"[CHECK] table tilt in robot base frame {tilt:.1f} deg; differs from {ref_name} "
                  f"by {dev:.1f} deg")
            if dev > args.max_table_tilt:
                msg = (f"The table differs by {dev:.1f} deg from {ref_name}: the camera calibration "
                       "does not match the robot (or the board moved). Run:  python run.py calibrate")
                if args.execute:
                    raise SystemExit(f"[ABORT] {msg}  (or --max_table_tilt if the table really is tilted)")
                print(f"[WARNING] {msg}")
        return grasps
    return run(args, plan_fn)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("gripper")
    g.add_argument("--cup_spacing", type=float, default=70.0, help="Centre distance of a 2-cup gripper, mm")
    g.add_argument("--cup_diameter", type=float, default=30.0, help="Cup diameter, mm")
    g.add_argument("--cups", help='Other layouts: cup centres in the tool XY plane, mm, e.g. "-35,0;35,0;0,40"')
    g.add_argument("--body_radius", type=float, default=45.0,
                   help="Radius of the suction tool body + flange + wrist for the collision check, mm")
    g.add_argument("--camera_radius", type=float, default=50.0,
                   help="Radius around the wrist camera for the collision check, mm")
    g.add_argument("--multi_object", action="store_true",
                   help="Allow one grasp to pick several parcels at once (the paper's mode). "
                        "Default: one parcel per pick")
    g.add_argument("--independent_cups", action="store_true",
                   help="Each cup has its own valve (allows grasps with only some cups on objects)")
    t = common.add_argument_group("tuning")
    t.add_argument("--voxel", type=float, default=5.0, help="Voxel size l, mm (paper: 5)")
    t.add_argument("--min_points", type=int, default=3, help="Points for an occupied voxel (paper: >10)")
    t.add_argument("--max_range", type=float, default=1200.0,
                   help="Ignore depth farther than this from the camera, mm (default 1200; 0 = off)")
    t.add_argument("--flatness", type=float, default=2.0, help="Max RMS unevenness under a cup, mm")
    t.add_argument("--min_height", type=float, default=8.0,
                   help="Ignore surfaces lower than this above the table; 0 = keep the table")
    t.add_argument("--max_tilt", type=float, default=45.0, help="Max surface / approach tilt, deg")
    common.add_argument("--frames", type=int, default=5, help="Depth frames to median-filter")
    common.add_argument("--show", action="store_true", help="Show the result window")
    common.add_argument("--bin_height", type=float, default=210.0,
                        help="Height of the virtual bin walls around the board, mm (default 100)")
    common.add_argument("--no_bin", action="store_true", help="Ignore the virtual bin walls (bin.json)")
    common.add_argument("--level_tol", type=float, default=5.0,
                        help="An object counts as tilted above this many degrees")

    sub.add_parser("demo", parents=[common], help="Synthetic scene with known answers")
    from suction_pick.robot import io_test
    io_test.add_arguments(sub.add_parser("io", help="Test a digital output, e.g. the suction (DO0) or release (DO1)"))
    sm = sub.add_parser("sim", parents=[common], help="RViz2 simulation of the planned pick (no robot)")
    sm.add_argument("--capture", help="pick_* capture folder to use (default: the newest)")
    sm.add_argument("--no_rviz", action="store_true", help="Only publish the topics, do not start RViz2")
    sm.add_argument("--fps", type=float, default=30.0, help="Animation frames per second")
    sm.add_argument("--approach_mm", type=float, default=60.0)
    sm.add_argument("--clearance_mm", type=float, default=30.0)
    sm.add_argument("--press_mm", type=float, default=3.0)
    sm.add_argument("--lift_mm", type=float, default=100.0)
    sm.add_argument("--arm_margin", type=float, default=30.0)
    sm.add_argument("--via_clearance", type=float, default=150.0)
    bn = sub.add_parser("bin", help="Measure the board outline -> virtual bin walls (robot does not move)")
    bn.add_argument("--replay", help="Measure from a saved pick_* capture instead of the live camera")
    bn.add_argument("--auto", action="store_true",
                    help="No window: take the automatic choice (largest surface fully in view)")
    bn.add_argument("--allow_partial", action="store_true",
                    help="Save even if a board edge is outside the camera view (wall placed where the view ends)")
    bn.add_argument("--frames", type=int, default=5)
    bn.add_argument("--max_range", type=float, default=2500.0,
                    help="Ignore depth beyond this, mm (the nearest large surface is taken as the table)")
    ca = sub.add_parser("calibrate", help="Camera rotation calibration from the board (moves the robot)")
    ca.add_argument("--tilt", type=float, default=15.0, help="Tilt of the calibration poses, deg (default 15)")
    ca.add_argument("--yes", action="store_true", help="Do not ask for confirmation before moving")
    ca.add_argument("--vel", type=float, default=10.0, help="Speed in %% (default 10)")
    ca.add_argument("--frames", type=int, default=5, help="Depth frames per pose")
    ca.add_argument("--max_range", type=float, default=1200.0, help="Ignore depth beyond this, mm")
    ca.add_argument("--suction_tool", type=int, default=SUCTION_TOOL)
    pl = sub.add_parser("plan", parents=[common], help="Live camera (or --replay), no robot")
    pl.add_argument("--replay", help="Folder with color.png, depth_mm.npy, K.npy, T_plan_cam.npy")
    pk = sub.add_parser("pick", parents=[common], help="Robot pick (dry run unless --execute)")
    pk.add_argument("--suction_tool", type=int, default=SUCTION_TOOL,
                    help=f"Fairino tool id of the suction gripper TCP (default {SUCTION_TOOL})")
    pk.add_argument("--cam_tool", type=int, default=CAM_TOOL,
                    help=f"Fairino tool id of the camera TCP (default {CAM_TOOL})")
    pk.add_argument("--cam_yaw", type=float, default=0.0, choices=(0.0, 90.0, 180.0, -90.0))
    pk.add_argument("--use_tool_rotation", action="store_true",
                    help="Trust the camera tool's Rx/Ry/Rz instead of the measured CAM_RPY_DEFAULT")
    pk.add_argument("--vacuum_do", type=int, nargs="+", default=[0],
                    help="Suction output(s): switched ON at the hover point (default DO0)")
    pk.add_argument("--release_do", type=int, default=1,
                    help="Release output: ON when the suction goes off, for --release_time (default DO1; -1 = none)")
    pk.add_argument("--release_time", type=float, default=2.0, help="Seconds the release output stays on (default 2)")
    pk.add_argument("--tool_io", action="store_true", help="Vacuum outputs are on the tool flange (SetToolDO)")
    pk.add_argument("--vacuum_di", type=int, help="Optional vacuum-sensor digital input (1 = vacuum OK)")
    pk.add_argument("--vacuum_wait", type=float, default=0.5, help="Seconds to wait after vacuum on/off")
    pk.add_argument("--approach_mm", type=float, default=60.0, help="Pre-grasp distance along the normal")
    pk.add_argument("--press_mm", type=float, default=3.0, help="Push this far past the surface to seal")
    pk.add_argument("--arm_margin", type=float, default=30.0,
                    help="Planned routes keep every arm link at least this far from the cubicle, mm")
    pk.add_argument("--via_clearance", type=float, default=150.0,
                    help="Via point (if needed) is this far above the wall tops, mm")
    pk.add_argument("--no_arm_check", action="store_true", help="Check only the end effector, not the whole arm")
    pk.add_argument("--fence_margin", type=float, default=10.0,
                    help="Live monitor stops the robot if any part comes this close to a virtual wall, mm")
    pk.add_argument("--clearance_mm", type=float, default=30.0,
                    help="Hover / lift at least this far above the highest obstacle near the path (bin rim)")
    pk.add_argument("--lift_mm", type=float, default=100.0, help="Lift straight up after gripping")
    pk.add_argument("--place", type=float, nargs=6, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                    help="Place pose of the suction TCP (base frame); omit to hold the object")
    pk.add_argument("--no_home", dest="home", action="store_false",
                    help="Do not return to the home pose after the pick")
    pk.add_argument("--no_start_home", dest="start_home", action="store_false",
                    help="Measure from the current pose instead of moving home first")
    pk.add_argument("--home_vel", type=float, default=20.0, help="Speed (%%) for moves to home")
    pk.add_argument("--max_table_tilt", type=float, default=3.0,
                    help="Refuse to move if the table differs more than this (deg) from the board measured at calibration")
    pk.add_argument("--execute", action="store_true", help="Move the robot (asks for 'y')")
    args = p.parse_args(argv)
    if args.command == "io":
        from suction_pick.robot import io_test
        return io_test.run(args)
    if args.command == "sim":
        from suction_pick.sim.rviz import run as sim
        return sim(args)
    if args.command == "bin":
        return cmd_bin(args)
    if args.command == "calibrate":
        from suction_pick.robot.calibrate import run as calibrate
        return calibrate(args)
    return {"demo": cmd_demo, "plan": cmd_plan, "pick": cmd_pick}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
