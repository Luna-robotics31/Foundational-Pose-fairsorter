"""Optional camera adapters. Neither adapter connects to the robot."""
from dataclasses import dataclass
from typing import Optional
import sys
import time

import cv2
import numpy as np


@dataclass
class Frame:
    color: np.ndarray
    depth_mm: Optional[np.ndarray] = None


class Webcam:
    def __init__(self, index):
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        self.metadata = {"source": "webcam", "camera_index": index, "depth_aligned_to_color": False,
                         "image_mirrored_by_app": False}
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                f"Cannot open webcam {index}. On macOS, run from your IDE's terminal or Terminal and allow that app under System Settings > Privacy & Security > Camera. "
                "An agent/background process may lack camera authorization even when the IDE has it. "
                "If permission is already enabled, close other camera apps or choose another --camera-index."
            )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self):
        ok, color = self.capture.read()
        return Frame(color) if ok else None

    def close(self):
        self.capture.release()


def _open_orbbec(ob, index, recover=True):
    """Open an Orbbec device; on Linux through the kernel V4L2 driver.

    The SDK's default LibUVC backend delivered no frames on the Gemini 2 here, and it
    detaches the kernel video driver so the next V4L2 open fails with
    "No v4l device found". That state is repaired by rebooting the camera.
    """
    context = ob.Context()
    context.set_logger_level(ob.OBLogLevel.ERROR)
    if sys.platform.startswith("linux"):
        context.set_uvc_backend_type(ob.OBUvcBackendType.V4L2)
    devices = context.query_devices()
    if index < 0 or index >= devices.get_count():
        raise RuntimeError(f"Orbbec camera index {index} unavailable; found {devices.get_count()} devices. Check USB, Linux udev rules, and close Orbbec Viewer/other camera apps.")
    try:
        return context, devices.get_device_by_index(index)
    except Exception as exc:
        if not recover or "No v4l device" not in str(exc):
            raise RuntimeError(f"Cannot open the Orbbec camera: {exc}. If it says busy (-6), another program (e.g. the cube tracker) is using it.") from exc
    print("Camera video interfaces unavailable (left detached by an earlier program); rebooting the camera...")
    del devices, context
    reset = ob.Context()  # automatic backend can still reach a half-attached camera
    reset.set_logger_level(ob.OBLogLevel.ERROR)
    reset.query_devices().get_device_by_index(index).reboot()
    del reset
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        time.sleep(1.0)
        try:
            return _open_orbbec(ob, index, recover=False)
        except Exception:
            continue
    raise RuntimeError("Camera did not come back after reboot; unplug and replug it.")


def _pick_profile(profiles, requests, what):
    """First available (width, height, format, fps) request."""
    for request in requests:
        try:
            return profiles.get_video_stream_profile(*request)
        except Exception:
            continue
    raise RuntimeError(f"No suitable {what} stream profile on this camera.")


class Orbbec:
    def __init__(self, index):
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise RuntimeError("Orbbec SDK is missing. Install requirements-camera.txt with this environment's Python; see README.") from exc
        self.ob = ob
        self.context, self.device = _open_orbbec(ob, index)
        self.pipeline = ob.Pipeline(self.device)
        config = ob.Config()
        color_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        # Prefer a modest profile suitable for live CPU testing. Fall back to
        # any RGB profile, then decode the device's default color format.
        try:
            color_profile = color_profiles.get_video_stream_profile(640, 480, ob.OBFormat.RGB, 30)
        except Exception:
            try:
                color_profile = color_profiles.get_video_stream_profile(0, 0, ob.OBFormat.RGB, 0)
            except Exception:
                color_profile = color_profiles.get_default_video_stream_profile()
        # Uncompressed Y16 depth. The Gemini 2's DEFAULT depth profile is 1280x800 RLE
        # (compressed): with it the depth-to-color alignment returned no frames at all.
        depth_profile = _pick_profile(
            self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR),
            [(640, 400, ob.OBFormat.Y16, 30), (0, 0, ob.OBFormat.Y16, 30), (0, 0, ob.OBFormat.Y16, 0)],
            "uncompressed (Y16) depth")
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self.aligner = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        info = self.device.get_device_info()
        self.metadata = {
            "source": "orbbec", "device": info.get_name(), "serial": info.get_serial_number(),
            "depth_aligned_to_color": True, "depth_unit": "mm",
            "alignment": "Orbbec AlignFilter to COLOR_STREAM; never a resize",
        }
        self.pipeline.enable_frame_sync()
        try:
            self.pipeline.start(config)
        except Exception as exc:
            raise RuntimeError(f"Cannot start Orbbec streams: {exc}. Close other camera apps and verify the device in Orbbec Viewer.") from exc

    def read(self):
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None
        aligned = self.aligner.process(frames)
        if aligned is None:
            return None
        frames = aligned.as_frame_set()
        color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None
        color = self._color_image(color_frame)
        raw_depth = np.frombuffer(depth_frame.get_data(), np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width())
        depth = raw_depth.astype(np.float32) * depth_frame.get_depth_scale()
        if color.shape[:2] != depth.shape:
            raise RuntimeError("SDK returned different color/depth sizes after alignment. Check stream profiles; resizing would not align the data.")
        # Copy: SDK buffers can be reused on the next frame; background history
        # and saved input pairs must own their arrays.
        return Frame(color.copy(), depth.copy())

    def _color_image(self, frame):
        ob = self.ob
        width, height, fmt = frame.get_width(), frame.get_height(), frame.get_format()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if fmt == ob.OBFormat.RGB:
            return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        if fmt == ob.OBFormat.BGR:
            return data.reshape(height, width, 3)
        if fmt == ob.OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Cannot decode the camera's MJPG color frame.")
            return image
        if fmt == ob.OBFormat.YUYV:
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
        raise RuntimeError(f"Unsupported camera color format: {fmt}. Select an RGB, BGR, MJPG, or YUYV profile.")

    def close(self):
        self.pipeline.stop()


def median_reference(frames):
    if not frames:
        raise ValueError("No camera frames available for an empty reference.")
    color = np.median(np.stack([frame.color for frame in frames]), axis=0).astype(np.uint8)
    depth = None
    if all(frame.depth_mm is not None for frame in frames):
        stack = np.stack([frame.depth_mm for frame in frames])
        invalid = ~np.isfinite(stack) | (stack <= 0)
        depth = np.ma.median(np.ma.array(stack, mask=invalid), axis=0).filled(0).astype(np.float32)
    return Frame(color, depth)
