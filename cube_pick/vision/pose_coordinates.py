"""Convert a FoundationPose object pose into camera and robot-frame coordinates."""

import numpy as np


def _rigid(transform, name):
    transform = np.asarray(transform, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-3)):
        raise ValueError(f"{name} must be a valid 4x4 rigid transform in metres")
    return transform


def read_transform(path):
    """Read a whitespace-separated 4x4 metric rigid transform."""
    return _rigid(np.loadtxt(path), str(path))


def pose_record(frame_id, T_camera_object, model_center,
                T_robot_camera=None, T_object_grasp=None):
    """Compose object and TCP targets; all translations use metres."""
    T_camera_object = _rigid(T_camera_object, "T_camera_object")
    model_center = np.asarray(model_center, dtype=float)
    if model_center.shape != (3,) or not np.isfinite(model_center).all():
        raise ValueError("model_center must be three finite coordinates in metres")
    center_camera = T_camera_object[:3, :3] @ model_center + T_camera_object[:3, 3]
    record = {"frame_id": str(frame_id), "units": "metres",
              "coordinate_frame": "camera_optical",
              "center_camera_m": center_camera.tolist(),
              "T_camera_object": T_camera_object.tolist()}
    if T_object_grasp is not None:
        T_object_grasp = _rigid(T_object_grasp, "T_object_grasp")
        T_camera_grasp = T_camera_object @ T_object_grasp
        record["T_camera_grasp"] = T_camera_grasp.tolist()
    if T_robot_camera is not None:
        T_robot_camera = _rigid(T_robot_camera, "T_robot_camera")
        record["robot_coordinate_frame"] = "robot_base"
        record["center_robot_m"] = (T_robot_camera[:3, :3] @ center_camera
                                    + T_robot_camera[:3, 3]).tolist()
        record["T_robot_object"] = (T_robot_camera @ T_camera_object).tolist()
        if T_object_grasp is not None:
            record["T_robot_grasp"] = (T_robot_camera @ T_camera_grasp).tolist()
    return record
