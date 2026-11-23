#!/usr/bin/env python3
"""Detect and track a mesh-based object with a live Orbbec RGB-D camera.

The camera aligns measured depth to rectified colour. A trained detector supplies
the object mask for FoundationPose registration and reacquisition. Q/Esc quits;
R resets tracking.

Run on its own:  python -m cube_pick.vision.live_tracker   (from the FoundationPose folder)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from cube_pick import config
from cube_pick.vision.pose_coordinates import pose_record, read_transform

# FoundationPose itself (estimater, Utils, mycpp) lives in the parent checkout.
config.add_to_path(config.FP_ROOT)
LOG = logging.getLogger("live_cube")


def depth_points(depth, mask, K, max_points=None):
    """Unproject valid depth pixels into the optical camera frame, in metres."""
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if max_points and len(u) > max_points:
        sampled = np.linspace(0, len(u) - 1, max_points, dtype=int)
        u, v = u[sampled], v[sampled]
    z = depth[v, u]
    return np.column_stack(((u - K[0, 2]) * z / K[0, 0],
                            (v - K[1, 2]) * z / K[1, 1], z))


def fit_background_plane(points, tolerance=0.004):
    """Fit the table/wall near a detection using a small deterministic RANSAC."""
    if len(points) < 80:
        return None
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(50):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        inliers = np.abs((points - a) @ normal) < tolerance
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(80, len(points) * 0.6):
        return None
    centre = points[best].mean(axis=0)
    _, singular_values, axes = np.linalg.svd(points[best] - centre, full_matrices=False)
    if singular_values[1] < 0.005:
        return None
    normal = axes[-1]
    return normal, -float(normal @ centre)


def automatic_mask(depth, box, K, extents, z_min=0.15, z_max=1.5, polygon=None):
    """Refine a detector box with depth and remove an observable support plane."""
    height, width = depth.shape
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    x0, x1 = np.clip([x0, x1], 0, width)
    y0, y1 = np.clip([y0, y1], 0, height)
    foreground = np.zeros(depth.shape, dtype=bool)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return foreground
    valid = np.isfinite(depth) & (depth >= z_min) & (depth <= z_max)
    foreground[y0:y1, x0:x1] = True
    if polygon is not None:
        silhouette = np.zeros(depth.shape, dtype=np.uint8)
        cv2.fillPoly(silhouette, [np.asarray(polygon, dtype=np.int32)], 1)
        foreground &= silhouette.astype(bool)

    centre = np.zeros(depth.shape, dtype=bool)
    dx, dy = (x1 - x0) // 4, (y1 - y0) // 4
    centre[y0 + dy:y1 - dy, x0 + dx:x1 - dx] = True
    seed_values = depth[centre & foreground & valid]
    if len(seed_values) < 10:
        return np.zeros(depth.shape, dtype=bool)
    seed_z = np.median(seed_values)
    foreground &= valid & (np.abs(depth - seed_z) < np.linalg.norm(extents) + 0.01)

    margin = max(12, int(max(x1 - x0, y1 - y0) * 0.6))
    surroundings = np.zeros(depth.shape, dtype=bool)
    surroundings[max(0, y0 - margin):min(height, y1 + margin),
                 max(0, x0 - margin):min(width, x1 + margin)] = True
    surroundings[y0:y1, x0:x1] = False
    plane = fit_background_plane(depth_points(depth, surroundings & valid, K, 800))
    if plane is not None:
        normal, offset = plane
        v, u = np.nonzero(foreground)
        points = depth_points(depth, foreground, K)
        keep = np.abs(points @ normal + offset) > max(0.005, float(min(extents)) * 0.1)
        foreground[v[~keep], u[~keep]] = False

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros(depth.shape, dtype=bool)
    chosen = max(range(1, count), key=lambda label: (
        np.count_nonzero((labels == label) & centre), stats[label, cv2.CC_STAT_AREA]))
    return labels == chosen


def support_plane(depth, box, K, z_min=0.15, z_max=1.5):
    """Fit the surface the object stands on, from a ring of depth around its box.

    Returns (unit normal, offset) in the camera frame with n·p + offset = 0 on the
    plane and the normal pointing toward the camera, so n·p + offset is the height of
    a point p above the surface. None when no dominant plane is visible.
    """
    height, width = depth.shape
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    x0, x1 = np.clip([x0, x1], 0, width)
    y0, y1 = np.clip([y0, y1], 0, height)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    margin = max(12, int(max(x1 - x0, y1 - y0) * 0.6))
    ring = np.zeros(depth.shape, dtype=bool)
    ring[max(0, y0 - margin):min(height, y1 + margin),
         max(0, x0 - margin):min(width, x1 + margin)] = True
    ring[y0:y1, x0:x1] = False
    valid = np.isfinite(depth) & (depth >= z_min) & (depth <= z_max)
    plane = fit_background_plane(depth_points(depth, ring & valid, K, 1500))
    if plane is None:
        return None
    normal, offset = plane
    if offset < 0:  # camera (origin) must be on the positive side
        normal, offset = -normal, -offset
    return normal, float(offset)


def height_above_plane(depth, mask, K, plane, percentile=95):
    """Object top height above the support plane, measured from its own depth pixels."""
    points = depth_points(depth, mask, K, 4000)
    if plane is None or len(points) < 50:
        return None
    normal, offset = plane
    return float(np.percentile(points @ normal + offset, percentile))


@dataclass
class Detection:
    box: np.ndarray
    confidence: float
    mask: np.ndarray
    plane: tuple | None = None          # support plane (normal, offset), camera frame
    top_height: float | None = None     # measured object top above that plane, metres


class CubeDetector:
    def __init__(self, weights, class_name, confidence, extents, z_min, z_max):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.classes = [int(i) for i, name in self.model.names.items()
                        if name.casefold() == class_name.casefold()]
        if not self.classes:
            raise ValueError(f"Class {class_name!r} not found in detector: {self.model.names}")
        self.confidence, self.extents = confidence, extents
        self.z_min, self.z_max = z_min, z_max
        LOG.info("Detector classes: %s; using %s", self.model.names, class_name)

    def detect(self, bgr, depth, K, previous_box=None):
        result = self.model.predict(bgr, conf=self.confidence, classes=self.classes,
                                    verbose=False)[0]
        if result.boxes is None:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        order = np.argsort(-scores)
        if previous_box is not None and len(boxes):
            lo = np.maximum(boxes[:, :2], previous_box[:2])
            hi = np.minimum(boxes[:, 2:], previous_box[2:])
            intersection = np.prod(np.maximum(hi - lo, 0), axis=1)
            area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            previous_area = np.prod(previous_box[2:] - previous_box[:2])
            iou = intersection / np.maximum(area + previous_area - intersection, 1)
            order = [i for i in np.argsort(-(iou + scores * 0.01)) if iou[i] > 0.05]
        for i in order:
            polygon = result.masks.xy[i] if result.masks is not None else None
            mask = automatic_mask(depth, boxes[i], K, self.extents,
                                  self.z_min, self.z_max, polygon)
            if np.count_nonzero(mask) >= 80:
                plane = support_plane(depth, boxes[i], K, self.z_min, self.z_max)
                return Detection(boxes[i], float(scores[i]), mask, plane,
                                 height_above_plane(depth, mask, K, plane))
        return None


def open_orbbec_device(ob, backend="V4L2", recover=True):
    """Open the first Orbbec device on a chosen Linux UVC backend.

    V4L2 streams through the kernel uvcvideo driver, so /dev/video* stays present while
    streaming and after exit (also for OpenCV scripts). The SDK's default LibUVC backend
    detaches that driver and, after an unclean exit, leaves some interfaces unbound:
    the video nodes vanish and the next V4L2 open fails ("No v4l device found").
    That state is repaired by rebooting the camera, which re-enumerates it on USB.
    """
    context = ob.Context()
    context.set_logger_level(ob.OBLogLevel.ERROR)
    context.set_uvc_backend_type(getattr(ob.OBUvcBackendType, backend))
    devices = context.query_devices()
    if devices.get_count() == 0:
        raise RuntimeError("No Orbbec camera found: check the USB 3 cable.")
    try:
        return context, devices.get_device_by_index(0)
    except Exception as exc:  # OBError: interfaces left unbound by an earlier LibUVC user
        if not recover:
            raise
        LOG.warning("Camera video interfaces unavailable (%s); rebooting the camera.", exc)
    del devices, context
    reset = ob.Context()  # automatic backend can still reach a half-attached camera
    reset.set_logger_level(ob.OBLogLevel.ERROR)
    reset.query_devices().get_device_by_index(0).reboot()
    del reset
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        time.sleep(1.0)
        try:
            return open_orbbec_device(ob, backend, recover=False)
        except Exception:
            continue
    raise RuntimeError("Camera did not come back after reboot; unplug and replug it.")


class OrbbecCamera:
    """Synchronised colour/depth registered and rectified to one colour camera."""

    def __init__(self, backend="V4L2"):
        import pyorbbecsdk as ob
        self.ob = ob
        self.context, self.device = open_orbbec_device(ob, backend)
        LOG.info("Camera: %s via %s backend", self.device.get_device_info().get_name(), backend)
        self.pipeline = ob.Pipeline(self.device)
        config = ob.Config()
        for sensor, width, height, fmt in (
            (ob.OBSensorType.COLOR_SENSOR, 640, 480, ob.OBFormat.RGB),
            (ob.OBSensorType.DEPTH_SENSOR, 640, 400, ob.OBFormat.Y16),
        ):
            profiles = self.pipeline.get_stream_profile_list(sensor)
            try:
                profile = profiles.get_video_stream_profile(width, height, fmt, 30)
            except RuntimeError:
                profile = profiles.get_default_video_stream_profile()
            config.enable_stream(profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self.pipeline.enable_frame_sync()
        self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        self.align.set_match_target_resolution(True)
        self.align.set_config_value("TargetDistortion", 1.0)
        self.K, self.maps = None, None
        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None
        frames = self.align.process(frames)
        if frames is None:
            return None
        frames = frames.as_frame_set()
        color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if color is None or depth_frame is None:
            return None
        received_ns = time.time_ns()
        width, height = color.get_width(), color.get_height()
        if (depth_frame.get_width(), depth_frame.get_height()) != (width, height):
            raise RuntimeError("Depth-to-colour registration failed; images must match without resizing.")
        data = np.frombuffer(color.get_data(), dtype=np.uint8)
        fmt, formats = color.get_format(), self.ob.OBFormat
        if fmt == formats.MJPG:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif fmt in (formats.RGB, formats.BGR):
            bgr = data.reshape(height, width, 3).copy()
            if fmt == formats.RGB:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        elif fmt in (formats.YUYV, formats.UYVY):
            conversion = (cv2.COLOR_YUV2BGR_YUYV if fmt == formats.YUYV
                          else cv2.COLOR_YUV2BGR_UYVY)
            bgr = cv2.cvtColor(data.reshape(height, width, 2), conversion)
        else:
            raise RuntimeError(f"Unsupported colour format: {fmt}; select an RGB/MJPG profile.")
        if bgr is None:
            return None
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
        depth = depth.astype(np.float32) * depth_frame.get_depth_scale() / 1000.0
        if self.K is None:
            profile = color.get_stream_profile().as_video_stream_profile()
            intr, dist = profile.get_intrinsic(), profile.get_distortion()
            if (intr.width, intr.height) != (width, height):
                raise RuntimeError("Colour intrinsics do not match the stream resolution.")
            self.K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=float)
            if not np.isfinite(self.K).all() or min(intr.fx, intr.fy) <= 0:
                raise RuntimeError("Invalid camera calibration from SDK.")
            model = dist.model.name
            if model not in ("NONE", "BROWN_CONRADY", "BROWN_CONRADY_K6"):
                raise RuntimeError(f"Unsupported lens distortion: {model}")
            coefficients = np.array([getattr(dist, name) for name in
                                     ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
            if not np.isfinite(coefficients).all():
                raise RuntimeError("Invalid distortion coefficients from SDK.")
            self.maps = cv2.initUndistortRectifyMap(
                self.K, coefficients, None, self.K, (width, height), cv2.CV_32FC1)
            LOG.info("Camera resolution %sx%s; calibrated K=%s", width, height, self.K.tolist())
        bgr = cv2.remap(bgr, *self.maps, cv2.INTER_LINEAR)
        depth = cv2.remap(depth, *self.maps, cv2.INTER_NEAREST)
        return bgr, depth, received_ns

    def stop(self):
        self.pipeline.stop()


def build_estimator(mesh, debug_dir):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose requires CUDA. Activate the GPU (CUDA) environment.")
    config.add_to_path(config.FP_ROOT / "mycpp/build")
    from estimater import FoundationPose
    from Utils import set_seed
    logging.getLogger().setLevel(logging.WARNING)
    set_seed(0)
    return FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                          mesh=mesh, debug=0, debug_dir=str(debug_dir))


class PoseTracker:
    """Register, track, and reject poses unsupported by the measured depth."""

    def __init__(self, estimator, mesh, register_iterations=5, track_iterations=2,
                 fit_distance=0.008, min_fit_fraction=0.55):
        self.estimator = estimator
        self.register_iterations, self.track_iterations = register_iterations, track_iterations
        self.fit_distance, self.min_fit_fraction = fit_distance, min_fit_fraction
        self.surface = cKDTree(trimesh.sample.sample_surface(mesh, 6000)[0])
        self.pose = None

    def reset(self):
        self.pose = None
        self.estimator.pose_last = None

    def update(self, rgb, depth, K, mask):
        if mask is None or np.count_nonzero(mask) < 80:
            self.reset()
            return None, {"reason": "Searching for cube with valid depth"}
        if self.pose is None:
            pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                           iteration=self.register_iterations)
        else:
            pose = self.estimator.track_one(rgb=rgb, depth=depth, K=K,
                                            iteration=self.track_iterations)
        pose = np.asarray(pose)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=0.02)
                or abs(np.linalg.det(pose[:3, :3]) - 1) > 0.02 or pose[2, 3] <= 0):
            self.reset()
            return None, {"reason": "Invalid pose; reacquiring"}
        points = depth_points(depth, mask, K, 500)
        local_points = (points - pose[:3, 3]) @ pose[:3, :3]
        distances, _ = self.surface.query(local_points)
        fraction = float(np.mean(distances < self.fit_distance)) if len(distances) else 0.0
        quality = {"depth_fit_fraction": fraction,
                   "median_surface_error_mm": float(np.median(distances) * 1000) if len(distances) else None}
        if fraction < self.min_fit_fraction:
            self.reset()
            return None, dict(quality, reason="Pose does not match depth; reacquiring")
        self.pose = pose
        return pose, dict(quality, reason="Tracking cube")


def obb_geometry(pose, bounds):
    """Return model-aligned box centre, dimensions and transformed corners."""
    centre, extents = bounds.mean(axis=0), np.diff(bounds, axis=0)[0]
    offset = np.eye(4)
    offset[:3, 3] = centre
    transform = pose @ offset
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    corners = (signs * extents / 2) @ transform[:3, :3].T + transform[:3, 3]
    return transform, extents, corners


def viewer_frame(display, K, pose, bounds, max_zoom=4.0):
    """Keep the full scene beside a crop that contains the posed box and axes."""
    height, width = display.shape[:2]
    if pose is None:
        return np.concatenate((display, display), axis=1)

    transform, extents, corners = obb_geometry(pose, bounds)
    axes = np.vstack((np.zeros(3), np.eye(3) * min(extents) * 0.8))
    axes = axes @ transform[:3, :3].T + transform[:3, 3]
    points = np.vstack((corners, axes))
    visible = points[:, 2] > 1e-6
    if not np.any(visible):
        return np.concatenate((display, display), axis=1)
    projected = points[visible] @ K.T
    uv = projected[:, :2] / projected[:, 2:]
    left, top = uv.min(axis=0)
    right, bottom = uv.max(axis=0)
    # A 60% margin keeps the whole object and its axes visible at the frame edge.
    box_width = max(right - left, 1.0) * 1.6
    box_height = max(bottom - top, 1.0) * 1.6
    crop_width = int(np.ceil(max(width / max_zoom, box_width,
                                 box_height * width / height)))
    crop_width = min(width, max(2, crop_width))
    crop_height = min(height, max(2, int(np.ceil(crop_width * height / width))))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    x0 = int(np.clip(round(center_x - crop_width / 2), 0, width - crop_width))
    y0 = int(np.clip(round(center_y - crop_height / 2), 0, height - crop_height))
    closeup = cv2.resize(display[y0:y0 + crop_height, x0:x0 + crop_width],
                         (width, height), interpolation=cv2.INTER_LINEAR)
    return np.concatenate((display, closeup), axis=1)


def write_result(output_dir, status, pose=None, bounds=None, detection=None,
                 quality=None, received_ns=None, T_robot_camera=None,
                 T_object_grasp=None):
    """Atomically replace current pose; invalid records contain no old pose."""
    record = {"valid": pose is not None, "status": status, "updated_unix_ns": time.time_ns(),
              "frame_received_unix_ns": received_ns, "coordinate_frame": "camera_optical",
              "axes": "X right, Y down, Z forward", "units": "metres",
              "quality": quality or {}}
    if pose is not None:
        transform, extents, corners = obb_geometry(pose, bounds)
        record.update(T_camera_object=np.asarray(pose).tolist(), T_camera_obb=transform.tolist(),
                      center_camera_m=transform[:3, 3].tolist(),
                      center_camera_mm=(transform[:3, 3] * 1000).tolist(),
                      extents_m=extents.tolist(), corners_camera_m=corners.tolist(),
                      detector_confidence=detection.confidence if detection is not None else None)
        if detection is not None and detection.plane is not None:
            # Measured from depth only (no mesh): the support surface and the object's
            # top above it. Lets a picker find table and object heights without hard-coding.
            normal, offset = detection.plane
            record["support_plane_camera"] = {"normal": np.asarray(normal).tolist(),
                                              "offset_m": float(offset)}
            record["object_top_height_m"] = detection.top_height
        if T_robot_camera is not None or T_object_grasp is not None:
            coordinates = pose_record(str(received_ns), pose, np.asarray(bounds).mean(axis=0),
                                      T_robot_camera, T_object_grasp)
            for key in ("robot_coordinate_frame", "center_robot_m", "T_robot_object",
                        "T_camera_grasp", "T_robot_grasp"):
                if key in coordinates:
                    record[key] = coordinates[key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "latest_pose.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output_dir / "latest_pose.json")
    return record


def draw_result(bgr, K, pose, bounds, detection, status, fps):
    display = bgr.copy()
    if detection is not None:
        display[detection.mask] = (display[detection.mask] * 0.8 + np.array([0, 40, 0])).clip(0, 255)
    if pose is not None:
        from Utils import draw_posed_3d_box, draw_xyz_axis
        transform, extents, _ = obb_geometry(pose, bounds)
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        display = draw_posed_3d_box(K, img=display, ob_in_cam=transform,
                                   bbox=np.stack((-extents / 2, extents / 2)))
        display = draw_xyz_axis(display, ob_in_cam=transform, scale=float(min(extents)) * 0.8,
                                K=K, thickness=2, is_input_rgb=True)
        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        xyz = transform[:3, 3] * 1000
        cv2.putText(display, f"Centre X {xyz[0]:.1f}  Y {xyz[1]:.1f}  Z {xyz[2]:.1f} mm",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if pose is not None else (0, 180, 255), 2)
    cv2.putText(display, f"{fps:.1f} fps | Q quit | R reacquire", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return display


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh_file", type=Path, default=config.MESH_FILE)
    parser.add_argument("--mesh_scale", type=float, default=1.0, help="OBJ units to metres")
    parser.add_argument("--weights", type=Path, default=config.DETECTOR_WEIGHTS)
    parser.add_argument("--class_name", default="white_cube")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--z_min", type=float, default=0.15)
    parser.add_argument("--z_max", type=float, default=1.5)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--fit_distance_mm", type=float, default=8.0)
    parser.add_argument("--min_fit_fraction", type=float, default=0.55)
    parser.add_argument("--debug_dir", type=Path, default=config.LIVE_DIR)
    parser.add_argument("--save_poses", action="store_true", help="Save each accepted object pose as text")
    parser.add_argument("--T_robot_camera", type=Path,
                        help="Measured camera optical to robot base 4x4 matrix, in metres")
    parser.add_argument("--T_object_grasp", type=Path,
                        help="Desired gripper TCP pose in OBJ coordinates, 4x4 in metres")
    parser.add_argument("--headless", action="store_true", help="Suppress OpenCV preview; Ctrl+C stops")
    parser.add_argument("--uvc_backend", choices=("V4L2", "LIBUVC", "AUTO"), default="V4L2",
                        help="Linux camera backend. V4L2 (default) keeps /dev/video* available "
                             "between runs; LIBUVC is the SDK's default and detaches it")
    parser.add_argument("--view_zoom", type=float, default=4.0,
                        help="Maximum cube close-up magnification in the preview (default: 4)")
    args = parser.parse_args(argv)
    for path in (args.mesh_file, args.weights):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if not (np.isfinite(args.mesh_scale) and args.mesh_scale > 0):
        parser.error("Mesh scale must be finite and positive.")
    if not np.isfinite(args.view_zoom) or args.view_zoom < 1:
        parser.error("View zoom must be finite and at least 1.")
    if not (0 < args.conf <= 1 and 0 < args.min_fit_fraction <= 1):
        parser.error("Confidence and minimum fit fraction must be in (0, 1].")
    if not (0 < args.z_min < args.z_max and args.fit_distance_mm > 0):
        parser.error("Require 0 < z_min < z_max and positive fit_distance_mm.")
    if min(args.est_refine_iter, args.track_refine_iter) < 1:
        parser.error("Refinement iteration counts must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    LOG.setLevel(logging.INFO)
    camera = None
    write_result(args.debug_dir, "initializing")
    try:
        T_robot_camera = read_transform(args.T_robot_camera) if args.T_robot_camera else None
        T_object_grasp = read_transform(args.T_object_grasp) if args.T_object_grasp else None
        mesh = trimesh.load(str(args.mesh_file), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("mesh_file must contain one triangle mesh.")
        mesh.apply_scale(args.mesh_scale)
        if not (np.isfinite(mesh.vertices).all() and np.all(mesh.extents > 0)):
            raise ValueError("Mesh has invalid vertices or dimensions.")
        if not 0.005 <= max(mesh.extents) <= 0.5:
            raise ValueError(f"Mesh dimensions {mesh.extents} are implausible for a cube in metres.")
        LOG.info("Mesh dimensions (mm): %s", np.round(mesh.extents * 1000, 2))
        estimator = build_estimator(mesh, args.debug_dir)
        detector = CubeDetector(args.weights, args.class_name, args.conf,
                                mesh.extents, args.z_min, args.z_max)
        tracker = PoseTracker(estimator, mesh, args.est_refine_iter, args.track_refine_iter,
                              args.fit_distance_mm / 1000, args.min_fit_fraction)
        camera = OrbbecCamera(args.uvc_backend)
        previous_box, last_print, retry_at = None, 0.0, 0.0
        window = "Automatic cube pose / OBB"
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        window_size = None
        LOG.info("Automatic acquisition started. Results: %s", args.debug_dir / "latest_pose.json")
        while True:
            started = time.monotonic()
            frame = camera.read()
            if frame is None:
                tracker.reset()
                previous_box = None
                write_result(args.debug_dir, "no_camera_frame")
                if not args.headless and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            bgr, depth, received_ns = frame
            if not args.headless and window_size != bgr.shape[:2]:
                height, width = bgr.shape[:2]
                viewer_width = min(1600, max(960, width * 2))
                cv2.resizeWindow(window, viewer_width,
                                 round(height * viewer_width / (width * 2)))
                window_size = bgr.shape[:2]
            detection = detector.detect(bgr, depth, camera.K, previous_box)
            pose, quality = None, {"reason": "Searching for cube with valid depth"}
            try:
                if time.monotonic() >= retry_at:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    pose, quality = tracker.update(rgb, depth, camera.K,
                                                   detection.mask if detection else None)
                    if pose is None and detection is not None:
                        retry_at = time.monotonic() + 0.5
            except (RuntimeError, ValueError) as exc:
                tracker.reset()
                retry_at = time.monotonic() + 0.5
                quality = {"reason": "Pose failed; reacquiring"}
                LOG.warning("Pose estimation failed: %s", exc)
            previous_box = detection.box if pose is not None else None
            status = quality["reason"]
            record = write_result(args.debug_dir, status, pose, mesh.bounds,
                                  detection, quality, received_ns,
                                  T_robot_camera, T_object_grasp)
            if pose is not None and args.save_poses:
                folder = args.debug_dir / "ob_in_cam"
                folder.mkdir(exist_ok=True)
                np.savetxt(folder / f"{received_ns}.txt", pose)
            if time.monotonic() - last_print >= 0.5:
                if record["valid"]:
                    LOG.info("Centre camera XYZ (mm): %s | depth fit %.0f%%",
                             np.round(record["center_camera_mm"], 1),
                             quality["depth_fit_fraction"] * 100)
                else:
                    LOG.info("%s", status)
                last_print = time.monotonic()
            fps = 1 / max(time.monotonic() - started, 1e-6)
            if not args.headless:
                overlay = draw_result(bgr, camera.K, pose, mesh.bounds, detection, status, fps)
                cv2.imshow(window, viewer_frame(overlay, camera.K, pose, mesh.bounds, args.view_zoom))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker.reset()
                    previous_box, retry_at = None, 0.0
                    write_result(args.debug_dir, "reacquiring")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        LOG.error("%s", exc)
        return 1
    finally:
        try:
            if camera is not None:
                camera.stop()
        finally:
            if not args.headless:
                cv2.destroyAllWindows()
            write_result(args.debug_dir, "stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
