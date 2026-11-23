"""RViz2 simulation of the suction pick: real planning code, no robot needed.

    python run.py sim                       # latest saved pick_* capture, opens RViz2
    python run.py sim --capture output/pick_20261119_165724 --bin_height 150

What it shows (topic /suction_sim/markers + /suction_sim/cloud, fixed frame base_link, metres):
  * the captured scene as a coloured point cloud (robot base frame, from the saved pick capture)
  * the board, the virtual walls (orange) and the stand below it (grey) = the cubicle
  * the real FR10 (robot/fr10/fr10.urdf: Fairino's V5 meshes on the controller's exact kinematics),
    published as /robot_description + TF and drawn by RViz's RobotModel display
  * see-through over it, the collision capsules of the whole-arm check, each link coloured by its
    distance to the cubicle: green > 50 mm, yellow 10-50, red < 10
  * the suction tool, cups and wrist camera (the end-effector collision model)
  * the planned pick, looping: home -> (via) -> hover -> descend -> suction -> lift with the
    parcel -> home -> release, with the phase and the closest distance as text
The grasp, route and clearances come from the same functions as `pick` (planner, plan_route);
only the controller IK is replaced by a numerical IK on the same FR10 kinematics.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from suction_pick.settings.paths import RVIZ_CONFIG, RVIZ_LOG  # written to output/
HOME_SEED = [-42.4, -104.1, -50.0, -133.6, 97.0, 13.8]   # real joint angles at the home pose


def _ensure_ros():
    """Re-run this command inside the ROS 2 environment if rclpy is not importable."""
    import importlib.util
    if importlib.util.find_spec("rclpy") is not None:
        return
    if os.environ.get("SUCTION_SIM_REEXEC"):
        raise SystemExit("rclpy not importable even after sourcing /opt/ros/humble/setup.bash")
    import shlex
    cmd = " ".join(shlex.quote(a) for a in [sys.executable, *sys.orig_argv[1:]])
    os.environ["SUCTION_SIM_REEXEC"] = "1"
    os.execvp("bash", ["bash", "-c", f"source /opt/ros/humble/setup.bash && exec {cmd}"])


class SimCell:
    """Stand-in for cube_pick's Cell: IK on the fitted FR10 model instead of the controller."""

    def __init__(self, T_flange_gripper, q_home):
        self.T_flange_gripper, self.q_home = T_flange_gripper, np.asarray(q_home, float)

    def reachable(self, gripper_pose, ref=None):
        from cube_pick.robot.pick_cube import pose_to_T
        from suction_pick.robot.arm import ik
        T = pose_to_T(gripper_pose) @ np.linalg.inv(self.T_flange_gripper)
        return ik(T, self.q_home if ref is None else ref)


# ── planning (same code as the real pick) ───────────────────────────────
def plan_scene(args):
    import cv2
    from cube_pick.config import SUCTION_TOOL, TOOL_COORDS
    from cube_pick.robot.home_pos import HOME_POSE
    from cube_pick.robot.pick_cube import pose_to_T
    from suction_pick.robot import arm
    from suction_pick.robot import workcell
    from suction_pick.app.cli import OUT_ROOT, run_pipeline, stamp
    from suction_pick.vision.geometry import depth_to_points, transform
    from suction_pick.robot.pick import equivalent_grasps, linear_chain_problem, plan_route, waypoints

    captures = sorted(OUT_ROOT.glob("pick_*"), key=lambda p: p.name)
    if not args.capture and not captures:
        raise SystemExit("[SIM] No saved scene yet. Capture one first with a dry run (the robot does not move):\n"
                         "      python run.py pick\n      then run:  python run.py sim")
    cap = Path(args.capture) if args.capture else captures[-1]
    print(f"[SIM] capture {cap}")
    bgr = cv2.imread(str(cap / "color.png"))
    depth, K, T_base_cam = np.load(cap / "depth_mm.npy"), np.load(cap / "K.npy"), np.load(cap / "T_plan_cam.npy")
    points = transform(T_base_cam, depth_to_points(depth, K))
    args.frame_name = "robot base frame"
    surface, grasps, _ = run_pipeline(bgr, depth, K, points, T_base_cam, args, cap / f"sim_{stamp()}")
    bin_ = None if args.no_bin else workcell.load()
    walls = workcell.arm_wall_points(bin_, args.bin_height) if bin_ is not None else None
    T_fg = pose_to_T(TOOL_COORDS[SUCTION_TOOL])
    q_home = arm.ik(pose_to_T(HOME_POSE), HOME_SEED)
    if q_home is None:
        raise SystemExit("[SIM] home pose has no IK solution near the recorded home joints")
    cell = SimCell(T_fg, q_home)
    chosen = None
    for i, grasp in enumerate(grasps):
        for g in equivalent_grasps(grasp):
            wps = waypoints(g, args)
            if linear_chain_problem(cell, [wps["pre-grasp"], wps["grasp"], wps["lift"]], ["pre-grasp", "grasp", "lift"]):
                continue
            route, report = plan_route(cell, wps, q_home, q_home, walls, args)
            print(f"[SIM] grasp #{i + 1}{' (turned 180)' if g is not grasp else ''}: "
                  + ("OK  " + " | ".join(report) if route else "rejected: " + report[0]))
            if route:
                chosen = (g, route)
                break
        if chosen:
            break
    return {"capture": cap, "bgr": bgr, "points": points, "surface": surface, "bin": bin_, "walls": walls,
            "q_home": q_home, "cell": cell, "chosen": chosen, "T_fg": T_fg}


def timeline(scene, args):
    """[(q, phase, attached)] frames of the planned pick."""
    from suction_pick.robot import arm
    g, w = scene["chosen"]
    cell, qh = scene["cell"], scene["q_home"]
    frames = []

    def add(qs, phase, attached):
        frames.extend((np.asarray(q, float), phase, attached) for q in qs)

    def lin(a, b, q0):
        return arm.linear_path(cell, a, b, q0, step_mm=4.0)

    q = qh
    add([qh] * 20, "home: measuring", False)
    if "via" in w:
        q_via = cell.reachable(w["via"])
        add(arm.joint_path(q, q_via, 70), "joint move to the via point (above the walls)", False)
        path = lin(w["via"], w["pre-grasp"], q_via)
    else:
        q_pre = cell.reachable(w["pre-grasp"])
        path = arm.joint_path(q, q_pre, 90)
    add(path, "joint move to the hover point" if "via" not in w else "down to the hover point", False)
    q = path[-1]
    add([q] * 12, "hover: suction ON (DO0)", False)
    path = lin(w["pre-grasp"], w["grasp"], q)
    add(path, "straight down onto the parcel", False)
    q = path[-1]
    add([q] * 15, "gripping", True)
    path = lin(w["grasp"], w["lift"], q)
    add(path, "lift (straight up along the normal)", True)
    q = path[-1]
    if "via" in w:
        path = lin(w["lift"], w["via"], q)
        add(path, "up to the via point", True)
        q = path[-1]
    add(arm.joint_path(q, qh, 90), "joint move home with the parcel", True)
    add([qh] * 25, "home: DO0 off + DO1 release (2 s)", False)
    return frames


# ── RViz markers ────────────────────────────────────────────────────────
def _quat_z_to(v):
    """Quaternion (x, y, z, w) rotating +Z onto v."""
    v = v / (np.linalg.norm(v) + 1e-12)
    c = np.cross([0, 0, 1.0], v)
    s = np.linalg.norm(c)
    if s < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if v[2] > 0 else (1.0, 0.0, 0.0, 0.0)
    ang = np.arctan2(s, v[2])
    c = c / s * np.sin(ang / 2)
    return (c[0], c[1], c[2], np.cos(ang / 2))


def _color(d):
    if d < 10:
        return (0.95, 0.1, 0.1, 0.95)
    if d < 50:
        return (1.0, 0.8, 0.0, 0.9)
    return (0.2, 0.85, 0.3, 0.85)


class MarkerFactory:
    def __init__(self, stamp_fn):
        from visualization_msgs.msg import Marker
        self.Marker, self.stamp_fn = Marker, stamp_fn

    def make(self, ns, mid, mtype, pos_mm, quat=(0, 0, 0, 1), scale_mm=(10, 10, 10), color=(1, 1, 1, 1), text=None):
        m = self.Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.stamp_fn()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, self.Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) / 1000 for v in pos_mm]
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = [float(v) for v in quat]
        m.scale.x, m.scale.y, m.scale.z = [max(float(v), 0.1) / 1000 for v in scale_mm]
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in color]
        if text is not None:
            m.text = text.replace(" ", "\u00a0")       # RViz2 text drops normal spaces
        return m

    def capsule(self, ns, mid, p0, p1, r, color):
        M = self.Marker
        mid_p, v = (p0 + p1) / 2, p1 - p0
        return [self.make(ns, mid, M.CYLINDER, mid_p, _quat_z_to(v), (2 * r, 2 * r, np.linalg.norm(v)), color),
                self.make(ns, mid + 1, M.SPHERE, p0, scale_mm=(2 * r,) * 3, color=color),
                self.make(ns, mid + 2, M.SPHERE, p1, scale_mm=(2 * r,) * 3, color=color)]


def static_markers(mf, scene, args):
    """Board, virtual walls, stand."""
    M = mf.Marker
    out = []
    b = scene["bin"]
    if b is None:
        return out
    C = b["corners_base"]
    z0 = float(C[:, 2].mean())
    centre = C.mean(axis=0)
    e0, e1 = C[1] - C[0], C[3] - C[0]
    yaw = np.arctan2(e0[1], e0[0])
    q = (0, 0, np.sin(yaw / 2), np.cos(yaw / 2))
    L0, L1 = np.linalg.norm(e0), np.linalg.norm(e1)
    out.append(mf.make("board", 0, M.CUBE, (centre[0], centre[1], z0 - 5), q, (L0, L1, 10), (0.85, 0.4, 0.2, 0.6)))
    out.append(mf.make("stand", 0, M.CUBE, (centre[0], centre[1], (z0 - 10) / 2), q, (L0, L1, max(z0 - 10, 1)),
                       (0.5, 0.5, 0.5, 0.25)))
    H = args.bin_height
    for i in range(4):
        a, c = C[i], C[(i + 1) % 4]
        mid = (a + c) / 2
        yaw_i = np.arctan2(*(c - a)[1::-1])
        out.append(mf.make("walls", i, M.CUBE, (mid[0], mid[1], z0 + H / 2), (0, 0, np.sin(yaw_i / 2), np.cos(yaw_i / 2)),
                           (np.linalg.norm(c - a), 8, H), (1.0, 0.5, 0.0, 0.35)))
    out.append(mf.make("label", 0, M.TEXT_VIEW_FACING, (centre[0], centre[1], z0 + H + 60), scale_mm=(1, 1, 35),
                       color=(1, 0.6, 0.1, 1), text=f"virtual bin {L0:.0f} x {L1:.0f} mm, walls {H:.0f} mm"))
    return out


def parcel_box(scene):
    """(T_base_parcel 4x4, size (3,)) of the grasped parcel from its region's points."""
    import cv2
    g, _ = scene["chosen"]
    labs = set(g.cup_labels[g.active].tolist())
    S = scene["surface"]
    pts = S.points[np.isin(S.labels, list(labs))]
    pts = pts[np.isfinite(pts).all(axis=1)]
    top = float(np.median(pts[:, 2]))
    bottom = float(scene["bin"]["corners_base"][:, 2].mean()) if scene["bin"] is not None else top - 50
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32))
    # region = the graspable top shrunk by a cup radius on each side: grow it back
    w, h = w + 30, h + 30
    T = np.eye(4)
    a = np.radians(ang)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = [cx, cy, (top + bottom) / 2]
    return T, np.array([w, h, max(top - bottom, 5)])


def robot_markers(mf, scene, q, phase, attached, T_grip_parcel, parcel_size, ee, params):
    from cube_pick.robot.pick_cube import T_to_pose
    from suction_pick.robot import arm
    from suction_pick.robot.fence import part_distances
    M = mf.Marker
    out, mid = [], 0
    walls = scene["walls"]
    worst = (np.inf, "-")
    link_d = {}
    for p0, p1, r, name in arm.capsules(q):              # the same check as the real pick
        d = float((arm.seg_distance(walls, p0, p1) - r).min()) if walls is not None else np.inf
        link_d[name] = min(link_d.get(name, np.inf), d)
        worst = min(worst, (d, name))
    for p0, p1, r, name in arm.capsules(q):              # see-through over the URDF meshes
        d = link_d[name]                                  # faint when clear, visible when close
        out += mf.capsule("arm collision capsules", mid, p0, p1, r, _color(d)[:3] + (0.08 if d >= 50 else 0.4,))
        mid += 3
    F = arm.fk_frames(q)
    T_tool = F[-1] @ scene["T_fg"]
    R_g = T_tool[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    tcp = T_tool[:3, 3]
    # end-effector model, coloured by its distance to the (virtual) walls
    d_ee = np.inf
    if walls is not None:
        name, d_ee, _ = min(part_distances((walls - tcp) @ R_g, ee, params.collision_tol), key=lambda t: t[1])
        worst = min(worst, (d_ee, name))
    col = _color(d_ee)
    (bx, by, _, _), (cx, cy, _, cz) = ee["columns"][0], ee["columns"][1]
    out += mf.capsule("ee", 0, tcp + R_g @ [bx, by, 15.0], F[-1][:3, 3], 40.0, col)
    cam = tcp + R_g @ [cx, cy, cz + 30.0]
    out.append(mf.make("ee", 10, M.CUBE, cam, scale_mm=(90, 30, 30), color=(0.3, 0.6, 1.0, 0.9)))
    for k, sgn in enumerate((-1, 1)):
        out.append(mf.make("ee", 20 + k, M.CYLINDER, tcp + R_g @ [sgn * 35.0, 0, 5.0], _quat_z_to(R_g[:, 2]),
                           (30, 30, 10), (0.1, 0.1, 0.1, 1)))
    if attached and T_grip_parcel is not None:
        T_tcp = np.eye(4)
        T_tcp[:3, :3], T_tcp[:3, 3] = R_g, tcp
        Tp = T_tcp @ T_grip_parcel
        from scipy.spatial.transform import Rotation
        qx, qy, qz, qw = Rotation.from_matrix(Tp[:3, :3]).as_quat()
        out.append(mf.make("parcel", 0, M.CUBE, Tp[:3, 3], (qx, qy, qz, qw), parcel_size, (0.95, 0.75, 0.3, 0.95)))
    else:
        out.append(mf.make("parcel", 0, M.CUBE, (0, 0, -5000), color=(0, 0, 0, 0)))
    txt = f"{phase}\nclosest to the cubicle: {worst[0]:.0f} mm ({worst[1]})" if np.isfinite(worst[0]) else phase
    out.append(mf.make("text", 0, M.TEXT_VIEW_FACING, F[3][:3, 3] + [0, 0, 250], scale_mm=(1, 1, 40),
                       color=(1, 1, 1, 1), text=txt))
    _ = T_to_pose
    return out, worst


def robot_description():
    """fr10.urdf with absolute file:// mesh paths (RViz loads the meshes from the repo)."""
    from suction_pick.robot.arm import FR10_DIR, URDF_FILE
    return URDF_FILE.read_text().replace('filename="meshes/', f'filename="file://{FR10_DIR}/meshes/')


def joint_tf(q, stamp):
    """TransformStamped for every URDF joint at joint angles q (deg)."""
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation
    from suction_pick.robot import arm
    out = []
    for parent, child, T in arm.joint_transforms(q):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (T[:3, 3] / 1000).tolist()
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
        out.append(t)
    return out


def cloud_msg(scene, header, hide_box=None):
    from sensor_msgs.msg import PointField
    from sensor_msgs_py import point_cloud2
    P = scene["points"]
    rgb = scene["bgr"]
    ok = np.isfinite(P).all(axis=2) & (np.nan_to_num(P[..., 2], nan=-1e9) > -500)
    pts, col = P[ok], rgb[ok]
    if hide_box is not None:                            # remove the parcel once it is picked up
        T, size = hide_box
        loc = (pts - T[:3, 3]) @ T[:3, :3]
        pts, col = pts[~np.all(np.abs(loc) < size / 2 + 5, axis=1)], col[~np.all(np.abs(loc) < size / 2 + 5, axis=1)]
    step = max(1, len(pts) // 60000)
    pts, col = pts[::step] / 1000.0, col[::step].astype(np.uint32)
    packed = (col[:, 2] << 16) | (col[:, 1] << 8) | col[:, 0]
    rgbf = packed.view(np.float32)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
    data = [(float(a), float(b), float(c), float(d)) for (a, b, c), d in zip(pts, rgbf)]
    return point_cloud2.create_cloud(header, fields, data)


def write_rviz_config(focus=(0.35, -0.55, 0.35), distance=2.6, yaw=2.4, path=RVIZ_CONFIG):
    fx, fy, fz = focus
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Cell Size: 0.1
      Plane Cell Count: 30
    - Class: rviz_default_plugins/PointCloud2
      Name: Scene (camera)
      Enabled: true
      Topic:
        Value: /suction_sim/cloud
      Style: Points
      Size (Pixels): 2
      Color Transformer: RGB8
    - Class: rviz_default_plugins/RobotModel
      Name: FR10 (URDF)
      Enabled: true
      Description Source: Topic
      Description Topic:
        Value: /robot_description
        Durability Policy: Transient Local
      Visual Enabled: true
      Collision Enabled: false
      Alpha: 1
    - Class: rviz_default_plugins/MarkerArray
      Name: Collision capsules, tool, bin, parcel
      Enabled: true
      Topic:
        Value: /suction_sim/markers
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: false
  Global Options:
    Fixed Frame: base_link
    Background Color: 40; 40; 48
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: %.2f
      Focal Point:
        X: %.3f
        Y: %.3f
        Z: %.3f
      Pitch: 0.45
      Yaw: %.3f
Window Geometry:
  Width: 1400
  Height: 900
""" % (distance, fx, fy, fz, yaw))
    return path


def run(args):
    _ensure_ros()
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TransformStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import Header, String
    from tf2_ros import TransformBroadcaster
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray
    from sensor_msgs.msg import PointCloud2
    from suction_pick.app.cli import build_config
    from suction_pick.planning.planner import ee_model

    scene = plan_scene(args)
    if scene["chosen"] is None:
        print("[SIM] No collision-free pick found for this capture; showing the scene only.")
    gripper, _, params = build_config(args)
    ee = ee_model(gripper, params.hover_radius)
    frames = timeline(scene, args) if scene["chosen"] else [(scene["q_home"], "no pick found", False)]
    T_grip_parcel, psize = None, None
    if scene["chosen"]:
        Tp, psize = parcel_box(scene)
        g, w = scene["chosen"]
        T_tcp = np.eye(4)
        T_tcp[:3, :3], T_tcp[:3, 3] = g.R, g.tcp
        T_grip_parcel = np.linalg.inv(T_tcp) @ Tp

    rclpy.init()
    node = Node("suction_sim")
    pub_m = node.create_publisher(MarkerArray, "/suction_sim/markers", 10)
    pub_c = node.create_publisher(PointCloud2, "/suction_sim/cloud", 1)
    tf = StaticTransformBroadcaster(node)
    tf_arm = TransformBroadcaster(node)
    pub_d = node.create_publisher(String, "/robot_description",
                                  QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    pub_d.publish(String(data=robot_description()))
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id, t.child_frame_id = "world", "base_link"
    t.transform.rotation.w = 1.0
    tf.sendTransform(t)
    mf = MarkerFactory(lambda: node.get_clock().now().to_msg())
    rviz = None
    if not args.no_rviz:
        # OpenCV's bundled Qt sets Qt plugin/font paths in this process; RViz must not inherit them
        # (it would load OpenCV's Qt plugins and exit).
        env = {k: v for k, v in os.environ.items() if not k.startswith(("QT_", "QTWEBENGINE"))}
        RVIZ_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = open(RVIZ_LOG, "w")
        focus, dist, yaw = (0.35, -0.55, 0.35), 2.6, 2.4
        if scene["bin"] is not None:                       # look at the robot base and the bin together
            c = scene["bin"]["corners_base"].mean(axis=0) / 1000.0
            focus = (c[0] / 2, c[1] / 2, 0.35)
            dist = max(2.0, 1.6 * float(np.hypot(c[0], c[1])) + 0.8)
            yaw = float(np.arctan2(c[1], c[0])) + np.pi / 2       # side view of the base -> bin line
        rviz = subprocess.Popen(["rviz2", "-d", str(write_rviz_config(focus, dist, yaw))], stdout=log,
                                stderr=subprocess.STDOUT, env=env)
        print("[SIM] RViz2 started (close it or Ctrl+C here to stop)")
    clear = MarkerArray()
    m = Marker()
    m.action = Marker.DELETEALL
    clear.markers.append(m)
    pub_m.publish(clear)
    statics = static_markers(mf, scene, args)
    worst_all = (np.inf, "-")
    print(f"[SIM] animating {len(frames)} frames (loops). Topics: /suction_sim/markers, /suction_sim/cloud")
    try:
        k, last_cloud, picked = 0, 0.0, False
        while rclpy.ok() and (rviz is None or rviz.poll() is None):
            q, phase, attached = frames[k]
            if k == 0:
                picked = False
            picked = picked or attached
            if time.monotonic() - last_cloud > 1.0 or k == 0:
                hdr = Header()
                hdr.frame_id, hdr.stamp = "base_link", node.get_clock().now().to_msg()
                pub_c.publish(cloud_msg(scene, hdr, (Tp, psize) if (picked and scene["chosen"]) else None))
                last_cloud = time.monotonic()
            tf_arm.sendTransform(joint_tf(q, node.get_clock().now().to_msg()))
            ms, worst = robot_markers(mf, scene, q, phase, attached, T_grip_parcel, psize, ee, params)
            worst_all = min(worst_all, worst)
            arr = MarkerArray()
            arr.markers = statics + ms
            pub_m.publish(arr)
            rclpy.spin_once(node, timeout_sec=0)
            time.sleep(1.0 / args.fps)
            k = (k + 1) % len(frames)
            if k == 0:
                print(f"[SIM] loop done: closest approach of the robot to the cubicle {worst_all[0]:.0f} mm "
                      f"({worst_all[1]})")
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if rviz is not None and rviz.poll() is None:
            rviz.terminate()
        try:                                              # a second Ctrl+C here must not print a traceback
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass
    return 0
