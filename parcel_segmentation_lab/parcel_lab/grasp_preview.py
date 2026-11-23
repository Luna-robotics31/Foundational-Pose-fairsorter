"""Two-cup proposals in IMAGE PIXELS, with no metric or robot pose claim.

Searches a segmented silhouette for a pair of circular footprints with good
boundary clearance. RGB cannot establish a vacuum seal, surface normal, height,
centre of mass, or collision-free 3D approach. This is a visualization prototype.
"""
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GraspSettings:
    # Illustrative scale only. Ratio matches 70 mm spacing / 30 mm diameter.
    cup_spacing_px: float = 84.0
    cup_radius_px: float = 18.0
    edge_margin_px: float = 4.0
    sample_step_px: int = 6

    def validate(self):
        values = (self.cup_spacing_px, self.cup_radius_px, self.edge_margin_px)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Preview dimensions must be finite pixel values.")
        if self.cup_spacing_px <= 0 or self.cup_radius_px <= 0 or self.edge_margin_px < 0:
            raise ValueError("Preview spacing/radius must be positive; margin cannot be negative.")
        if self.sample_step_px < 1:
            raise ValueError("Preview sample step must be at least one pixel.")


@dataclass
class Proposal:
    instance_id: int
    status: str
    reason: Optional[str] = None
    midpoint_xy_px: Optional[Tuple[float, float]] = None
    cup_centres_xy_px: Optional[List[Tuple[float, float]]] = None
    cup_axis_angle_deg: Optional[float] = None
    layout_corners_xy_px: Optional[List[Tuple[float, float]]] = None
    boundary_clearance_px: Optional[float] = None
    ranking_score: Optional[float] = None


@dataclass
class Preview:
    settings: GraspSettings
    proposals: List[Proposal]
    target_instance_id: Optional[int]
    coordinate_system: str = "Original unmirrored image: u increases right, v increases down. Angle is clockwise from image-right, modulo 180 degrees."
    scale_calibrated: bool = False
    robot_pose: None = None
    method: str = "2D silhouette clearance search; not FoundationPose or a verified suction grasp"
    reported_cup_spacing_mm: float = 70.0
    reported_cup_diameter_range_mm: Tuple[float, float] = (20.0, 30.0)


def _proposal(instance, settings):
    if instance.touches_roi_boundary:
        return Proposal(instance.id, "rejected", "Parcel mask touches the image/ROI boundary; use a complete mask.")
    if settings.cup_spacing_px < 2 * settings.cup_radius_px:
        return Proposal(instance.id, "rejected", "Preview cup circles overlap. Increase spacing or reduce radius.")
    x, y, w, h = instance.bbox_xywh
    crop = instance.mask[y:y + h, x:x + w]
    # Padding makes the distance to an image/crop boundary well-defined.
    padded = np.pad(crop, 1, constant_values=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    required = math.ceil(settings.cup_radius_px + settings.edge_margin_px) + 1
    yy, xx = np.mgrid[0:h:settings.sample_step_px, 0:w:settings.sample_step_px]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    points = points[crop[points[:, 1], points[:, 0]] > 0].astype(np.float32)
    # Also sample the centre and deepest interior point to avoid a grid offset
    # unnecessarily missing small, otherwise feasible contact regions.
    deepest = np.unravel_index(int(np.argmax(distance)), distance.shape)
    centroid = np.array(instance.centroid_xy_px, np.float32) - (x, y)
    extra = np.array([(deepest[1], deepest[0]), centroid], np.float32)
    extra = np.rint(extra).astype(np.int32)
    inside = (extra[:, 0] >= 0) & (extra[:, 0] < w) & (extra[:, 1] >= 0) & (extra[:, 1] < h)
    extra = extra[inside]
    extra = extra[crop[extra[:, 1], extra[:, 0]] > 0]
    points = np.vstack((points, extra)).astype(np.float32)
    if not len(points):
        return Proposal(instance.id, "rejected", "No interior midpoint candidates.")

    box = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(crop)))
    edges = np.roll(box, -1, axis=0) - box
    major = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    principal = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180
    angles = [principal] + [float(angle) for angle in range(0, 180, 10)]
    best = None
    centering_cost = 0.025 * np.linalg.norm(points - centroid, axis=1)
    for angle in angles:
        radians = math.radians(angle)
        direction = np.array([math.cos(radians), math.sin(radians)], np.float32)
        offset = settings.cup_spacing_px * 0.5 * direction
        first, second = points - offset, points + offset
        a, b = np.rint(first).astype(np.int32), np.rint(second).astype(np.int32)
        in_bounds = ((a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
                     & (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h))
        indices = np.flatnonzero(in_bounds)
        if not len(indices):
            continue
        minimum_distance = np.minimum(distance[a[indices, 1], a[indices, 0]],
                                      distance[b[indices, 1], b[indices, 0]])
        fits = minimum_distance >= required
        indices, minimum_distance = indices[fits], minimum_distance[fits]
        if not len(indices):
            continue
        scores = minimum_distance - settings.cup_radius_px - settings.edge_margin_px - centering_cost[indices]
        local = int(np.argmax(scores))
        score = float(scores[local])
        if best is None or score > best[0] + 1e-5:
            index = indices[local]
            best = (score, angle, points[index], first[index], second[index], float(minimum_distance[local]))
    if best is None:
        return Proposal(instance.id, "rejected", "Both cup circles plus margin do not fit within this mask at the sampled poses.")

    score, angle, centre, first, second, clearance = best
    origin = np.array([x, y], np.float32)
    centre, first, second = centre + origin, first + origin, second + origin
    radians = math.radians(angle)
    along = np.array([math.cos(radians), math.sin(radians)], np.float32)
    across = np.array([-along[1], along[0]], np.float32)
    reach = settings.cup_spacing_px / 2 + settings.cup_radius_px
    corners = [centre + a * reach * along + b * settings.cup_radius_px * across
               for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return Proposal(
        instance.id, "candidate_2d", midpoint_xy_px=tuple(float(v) for v in centre),
        cup_centres_xy_px=[tuple(float(v) for v in point) for point in (first, second)],
        cup_axis_angle_deg=float(angle),
        layout_corners_xy_px=[tuple(float(v) for v in point) for point in corners],
        boundary_clearance_px=clearance - settings.cup_radius_px,
        ranking_score=score,
    )


def propose(result, settings):
    settings.validate()
    proposals = [_proposal(instance, settings) for instance in result.instances]
    accepted = [item for item in proposals if item.status == "candidate_2d"]
    target = max(accepted, key=lambda item: item.ranking_score).instance_id if accepted else None
    return Preview(settings, proposals, target)


def draw_preview(overlay, preview):
    """Add a projected layout and compact data panel to a segmentation overlay."""
    image = overlay.copy()
    selected = None
    for item in preview.proposals:
        if item.status != "candidate_2d" or item.instance_id != preview.target_instance_id:
            continue
        # Full-parcel boxes and labels are already drawn in the base overlay.
        # Show a small cup layout only on the selected parcel, in white.
        color = (255, 255, 255)
        thickness = 2
        corners = np.rint(item.layout_corners_xy_px).astype(np.int32)
        cv2.polylines(image, [corners], True, color, thickness, cv2.LINE_AA)
        for point in corners:
            cv2.rectangle(image, tuple(point - 3), tuple(point + 3), color, thickness)
        cups = [tuple(int(round(v)) for v in point) for point in item.cup_centres_xy_px]
        for point in cups:
            cv2.circle(image, point, int(round(preview.settings.cup_radius_px)), color, thickness, cv2.LINE_AA)
        cv2.arrowedLine(image, cups[0], cups[1], color, thickness, cv2.LINE_AA, tipLength=0.16)
        midpoint = tuple(int(round(v)) for v in item.midpoint_xy_px)
        cv2.drawMarker(image, midpoint, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        selected = item
    width, height = image.shape[1], image.shape[0]
    cv2.rectangle(image, (0, 0), (width - 1, 31), (20, 20, 20), -1)
    cv2.putText(image, "TOP VIEW | PURPLE = target | WHITE = cup layout | PIXELS", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 110), 1, cv2.LINE_AA)
    if selected is not None:
        u, v = selected.midpoint_xy_px
        lines = [f"P{selected.instance_id}: midpoint u={u:.0f}, v={v:.0f} px | cup axis {selected.cup_axis_angle_deg:.1f} deg CW",
                 f"Spacing {preview.settings.cup_spacing_px:g}px | diameter {2 * preview.settings.cup_radius_px:g}px | 180 deg symmetric",
                 "Depth / surface normal / robot XYZ-RPY: unavailable"]
    else:
        lines = ["No complete two-cup layout fits the current masks.",
                 "Check segmentation, ROI, spacing and cup size.",
                 "Depth / surface normal / robot XYZ-RPY: unavailable"]
    panel_top = height
    image = cv2.copyMakeBorder(image, 0, 82, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    for index, text in enumerate(lines):
        cv2.putText(image, text, (8, panel_top + 19 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (255, 255, 255), 1, cv2.LINE_AA)
    return image
