"""Offline experiments and an interactive camera preview."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import numpy as np

from .artifacts import draw_overlay, read_color, read_depth, save_run, write_image
from .demo import write_demo_inputs
from .grasp_preview import GraspSettings, draw_preview, propose
from .segmentation import Settings, segment
from .sources import Orbbec, Webcam, median_reference


def add_settings(parser):
    parser.add_argument("--mode", choices=("rgb", "depth", "hybrid"), default=None)
    parser.add_argument("--threshold", type=float, default=25, help="RGB Lab difference threshold (default 25).")
    parser.add_argument("--depth-change-mm", type=float, default=15, help="Minimum distance closer than empty reference, in mm (default 15).")
    parser.add_argument("--min-area", type=int, default=1000, help="Ignore components smaller than this pixel area.")
    parser.add_argument("--open-kernel", type=int, default=3, help="Odd morphology opening size; 0 disables it.")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Restrict processing to the bin interior, in original image pixels.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Parent directory for unique result folders.")
    parser.add_argument("--grasp-preview", action="store_true", help="Highlight a two-cup IMAGE-SPACE candidate and its angle; no robot pose.")
    parser.add_argument("--cup-spacing-px", type=float, default=84, help="Illustrative cup centre spacing in pixels (not 70 mm converted to pixels).")
    parser.add_argument("--cup-radius-px", type=float, default=18, help="Illustrative cup radius in pixels.")
    parser.add_argument("--edge-margin-px", type=float, default=4, help="Additional image-mask clearance around each cup.")


def grasp_settings_for(args):
    settings = GraspSettings(args.cup_spacing_px, args.cup_radius_px, args.edge_margin_px)
    settings.validate()
    return settings


def overlay_for(color, result, roi, preview):
    overlay = draw_overlay(color, result, roi, preview.target_instance_id if preview is not None else None)
    return draw_preview(overlay, preview) if preview is not None else overlay


def settings_for(args, default_mode):
    settings = Settings(args.mode or default_mode, args.threshold, args.depth_change_mm,
                        args.min_area, args.open_kernel, tuple(args.roi) if args.roi else None)
    settings.validate()
    return settings


def print_result(result, directory):
    print(f"Found {len(result.instances)} foreground candidate(s). Results: {directory.resolve()}")
    for message in result.warnings:
        print(f"Note: {message}")


def show_image(image):
    name = "Parcel segmentation | any key to close"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, min(1280, image.shape[1]), min(800, image.shape[0]))
    try:
        cv2.imshow(name, image)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


def run_image(args):
    if args.expected_count is not None and args.expected_count < 0:
        raise ValueError("Expected count cannot be negative.")
    color, background = read_color(args.image), read_color(args.background)
    depth = read_depth(args.depth) if args.depth else None
    background_depth = read_depth(args.background_depth) if args.background_depth else None
    settings = settings_for(args, "depth" if depth is not None else "rgb")
    if args.select_roi:
        selected = cv2.selectROI("Select bin interior, then ENTER", color, showCrosshair=True)
        cv2.destroyAllWindows()
        if selected[2] <= 0 or selected[3] <= 0:
            raise ValueError("ROI selection cancelled.")
        settings.roi = tuple(int(v) for v in selected)
    result = segment(color, background, settings, depth, background_depth)
    preview = propose(result, grasp_settings_for(args)) if args.grasp_preview else None
    metadata = {"scene_path": str(args.image.resolve()), "background_path": str(args.background.resolve())}
    if args.expected_count is not None:
        metadata.update(expected_count=args.expected_count, count_matches=len(result.instances) == args.expected_count)
    directory = save_run(args.output, color, background, result, settings, depth,
                         background_depth, metadata, grasp_preview=preview)
    print_result(result, directory)
    if args.show:
        show_image(overlay_for(color, result, settings.roi, preview))
    if args.expected_count is not None and len(result.instances) != args.expected_count:
        print(f"FAIL: expected {args.expected_count} parcels, found {len(result.instances)} components. Inspect the masks; count alone does not establish correct segmentation.")
        return 1
    return 0


def run_demo(args):
    color, background, depth, background_depth, labels = write_demo_inputs(args.data_dir)
    modes = [args.mode] if args.mode else ["rgb", "depth", "hybrid"]
    passed = True
    for mode in modes:
        settings = settings_for(args, mode)
        result = segment(color, background, settings, depth, background_depth)
        preview = propose(result, grasp_settings_for(args)) if args.grasp_preview else None
        truth = labels > 0
        predicted = result.foreground > 0
        iou = float(np.count_nonzero(truth & predicted) / np.count_nonzero(truth | predicted))
        ok = len(result.instances) == 3 and iou >= 0.95
        metadata = {"synthetic_demo": True, "expected_count": 3, "foreground_iou": iou, "passed": ok}
        directory = save_run(args.output, color, background, result, settings, depth,
                             background_depth, metadata, prefix=f"demo_{mode}", grasp_preview=preview)
        print_result(result, directory)
        print(f"{'PASS' if ok else 'FAIL'} {mode}: expected 3, found {len(result.instances)}; foreground IoU={iou:.4f}")
        passed = passed and ok
        if args.show:
            show_image(overlay_for(color, result, settings.roi, preview))
    print(f"Artificial input images saved to {args.data_dir.resolve()}. Real parcel accuracy needs real images.")
    return 0 if passed else 1


def save_reference(root, reference, metadata):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / f"reference_{stamp}_{uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    write_image(directory / "background.png", reference.color)
    if reference.depth_mm is not None:
        np.save(directory / "background_depth_mm.npy", reference.depth_mm, allow_pickle=False)
    (directory / "camera.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def run_live(args):
    if args.background_frames < 1 or args.background_frames > 31:
        raise ValueError("Background frames must be between 1 and 31.")
    settings = settings_for(args, "depth" if args.source == "orbbec" else "rgb")
    grasp_settings = grasp_settings_for(args) if args.grasp_preview else None
    if args.source == "webcam" and settings.mode != "rgb":
        raise ValueError("Webcam provides color only; use --mode rgb or --source orbbec.")
    source = None
    window = "Parcel lab | B: empty reference | R: ROI | F: freeze | S: save | Q: quit"
    history = deque(maxlen=args.background_frames)
    reference = None
    last_frame_time = time.monotonic()
    displayed = False
    frozen = False
    frame = None
    if args.max_frames < 0:
        raise ValueError("Maximum frames cannot be negative.")
    frame_count = 0
    try:
        source = Orbbec(args.camera_index) if args.source == "orbbec" else Webcam(args.camera_index)
        print(json.dumps(source.metadata, indent=2))
        print("Keep the camera stationary. With the work area EMPTY and hands clear, press B. Add separated parcels. R selects the area; F freezes; S saves; Q exits.")
        print("B uses the last background-frames frames. Hold the empty scene steady before pressing it. Recapture after any camera movement.")
        if grasp_settings is not None:
            print("Green boxes = full parcels; purple box = selected target; white circles/cross = 2D cup layout/midpoint. Angle = clockwise from image-right, modulo 180. Pixel spacing is UNCALIBRATED; no surface normal, robot XYZ, or seal verification.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 620)
        cv2.createTrackbar("RGB threshold", window, max(1, int(settings.color_threshold)), max(300, int(settings.color_threshold)), lambda _: None)
        cv2.createTrackbar("Depth change mm", window, max(1, int(settings.min_depth_change_mm)), max(200, int(settings.min_depth_change_mm)), lambda _: None)
        cv2.createTrackbar("Minimum area px", window, settings.min_area, max(20000, settings.min_area), lambda _: None)
        if grasp_settings is not None:
            cv2.createTrackbar("Cup spacing px", window, int(grasp_settings.cup_spacing_px), max(400, int(grasp_settings.cup_spacing_px)), lambda _: None)
            cv2.createTrackbar("Cup radius px", window, int(grasp_settings.cup_radius_px), max(150, int(grasp_settings.cup_radius_px)), lambda _: None)
        while True:
            if not frozen:
                frame = source.read()
            if frame is None:
                if time.monotonic() - last_frame_time > 10:
                    raise RuntimeError("No usable camera frames for 10 seconds. Check USB and close other camera applications.")
                if not displayed:  # instead of an empty black window
                    waiting = np.full((480, 640, 3), 30, np.uint8)
                    cv2.putText(waiting, "Waiting for camera frames...", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window, waiting)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_frame_time = time.monotonic()
            if not frozen:
                history.append(frame)
                frame_count += 1
            settings.color_threshold = max(1, cv2.getTrackbarPos("RGB threshold", window))
            settings.min_depth_change_mm = max(1, cv2.getTrackbarPos("Depth change mm", window))
            settings.min_area = max(1, cv2.getTrackbarPos("Minimum area px", window))
            if grasp_settings is not None:
                grasp_settings.cup_spacing_px = max(1, cv2.getTrackbarPos("Cup spacing px", window))
                grasp_settings.cup_radius_px = max(1, cv2.getTrackbarPos("Cup radius px", window))
            result = None
            grasp_preview = None
            if reference is None:
                preview = frame.color.copy()
                message = f"EMPTY bin, hands clear: press B ({len(history)}/{args.background_frames} reference frames ready)"
            else:
                result = segment(frame.color, reference.color, settings, frame.depth_mm, reference.depth_mm)
                if grasp_settings is not None:
                    grasp_preview = propose(result, grasp_settings)
                overlay = overlay_for(frame.color, result, settings.roi, grasp_preview)
                mask_view = cv2.cvtColor(result.foreground, cv2.COLOR_GRAY2BGR)
                mask_view = cv2.copyMakeBorder(mask_view, 0, overlay.shape[0] - mask_view.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                preview = np.hstack((overlay, mask_view))
                message = f"Mode: {settings.mode} | {len(result.instances)} candidates | S saves this frame"
                if result.warnings:
                    message += " | Check warnings in saved JSON"
            if frozen:
                message = "FROZEN | " + message
            preview = cv2.copyMakeBorder(preview, 0, 28, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(preview, message, (8, preview.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, preview)
            displayed = True
            key = cv2.waitKey(20 if frozen else 1) & 0xFF
            if args.max_frames and frame_count >= args.max_frames:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if displayed and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("b"), ord("B")):
                if frozen:
                    print("Unfreeze with F before capturing the empty reference.")
                    continue
                if len(history) < args.background_frames:
                    print("Wait until the reference frame buffer is full, then press B again.")
                    continue
                reference = median_reference(list(history))
                path = save_reference(args.output, reference, source.metadata)
                print(f"Empty reference captured: {path.resolve()}")
            elif key in (ord("r"), ord("R")):
                selected = cv2.selectROI("Select bin interior, then ENTER", frame.color, showCrosshair=True)
                cv2.destroyWindow("Select bin interior, then ENTER")
                if selected[2] > 0 and selected[3] > 0:
                    settings.roi = tuple(int(v) for v in selected)
                    print(f"ROI set to x y width height: {settings.roi}")
            elif key in (ord("f"), ord("F")):
                frozen = not frozen
            elif key in (ord("s"), ord("S")):
                if result is None:
                    print("Capture the empty reference with B first.")
                    continue
                directory = save_run(args.output, frame.color, reference.color, result,
                                     settings, frame.depth_mm, reference.depth_mm, source.metadata,
                                     grasp_preview=grasp_preview)
                print_result(result, directory)
    finally:
        if source is not None:
            source.close()
        cv2.destroyAllWindows()
    return 0


def make_parser():
    parser = argparse.ArgumentParser(description="Test segmentation of separated parcels using a stationary camera and an empty reference. No training or robot connection.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate and verify an artificial three-parcel scene.")
    add_settings(demo)
    demo.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    demo.add_argument("--show", action="store_true", help="Open a desktop preview.")
    demo.set_defaults(handler=run_demo)
    image = commands.add_parser("image", help="Segment a saved scene against an empty reference.")
    add_settings(image)
    image.add_argument("--image", type=Path, required=True)
    image.add_argument("--background", type=Path, required=True)
    image.add_argument("--depth", type=Path, help="Aligned scene depth .npy or uint16 PNG, in mm.")
    image.add_argument("--background-depth", type=Path, help="Aligned empty reference depth, in mm.")
    image.add_argument("--expected-count", type=int, help="Return exit code 1 if component count differs.")
    image.add_argument("--show", action="store_true")
    image.add_argument("--select-roi", action="store_true", help="Select the bin interior with the mouse (desktop required).")
    image.set_defaults(handler=run_image)
    live = commands.add_parser("live", help="Interactive MacBook/webcam RGB or Gemini 2 RGB-D preview.")
    add_settings(live)
    live.add_argument("--source", choices=("orbbec", "webcam"), default="webcam")
    live.add_argument("--camera-index", type=int, default=0)
    live.add_argument("--background-frames", type=int, default=9)
    live.add_argument("--max-frames", type=int, default=0, help="Exit after this many acquired frames; 0 keeps the preview open.")
    live.set_defaults(handler=run_live)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
