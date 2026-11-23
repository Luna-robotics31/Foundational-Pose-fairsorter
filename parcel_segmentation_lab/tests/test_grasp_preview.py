"""Check projected two-cup geometry without claiming a physical grasp."""
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from parcel_lab.artifacts import save_run
from parcel_lab.cli import main
from parcel_lab.demo import make_scene
from parcel_lab.grasp_preview import GraspSettings, propose
from parcel_lab.segmentation import Instance, Result, Settings, segment


def result_for(mask, clipped=False):
    moments = cv2.moments(mask, binaryImage=True)
    item = Instance(1, mask, cv2.boundingRect(mask), int(np.count_nonzero(mask)),
                    (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                    clipped, None, None)
    return Result(mask, [item], [])


def axis_error(first, second):
    return abs((first - second + 90) % 180 - 90)


class GraspPreviewTests(unittest.TestCase):
    def setUp(self):
        self.settings = GraspSettings()

    def assert_circles_inside(self, mask, item):
        radius = math.ceil(self.settings.cup_radius_px + self.settings.edge_margin_px)
        for point in item.cup_centres_xy_px:
            footprint = np.zeros_like(mask)
            centre = tuple(int(round(v)) for v in point)
            cv2.circle(footprint, centre, radius, 255, -1)
            self.assertTrue(np.all(mask[footprint > 0] > 0))
        u, v = (int(round(value)) for value in item.midpoint_xy_px)
        self.assertGreater(mask[v, u], 0)

    def test_rotated_narrow_parcel_has_matching_cup_axis_and_contained_circles(self):
        mask = np.zeros((400, 400), np.uint8)
        polygon = np.rint(cv2.boxPoints(((200, 200), (240, 64), 32))).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        preview = propose(result_for(mask), self.settings)
        item = preview.proposals[0]
        self.assertEqual(item.status, "candidate_2d")
        self.assertLess(axis_error(item.cup_axis_angle_deg, 32), 3)
        self.assertAlmostEqual(np.linalg.norm(np.subtract(*item.cup_centres_xy_px)), 84, places=3)
        self.assert_circles_inside(mask, item)

    def test_concave_mask_does_not_use_centroid_outside_the_parcel(self):
        mask = np.zeros((400, 400), np.uint8)
        mask[40:340, 40:140] = 255
        mask[240:340, 40:340] = 255
        result = result_for(mask)
        cx, cy = (int(round(v)) for v in result.instances[0].centroid_xy_px)
        self.assertEqual(mask[cy, cx], 0)
        item = propose(result, self.settings).proposals[0]
        self.assertEqual(item.status, "candidate_2d")
        self.assert_circles_inside(mask, item)

    def test_narrow_parcel_has_no_two_cup_fit(self):
        mask = np.zeros((300, 300), np.uint8)
        mask[100:130, 30:260] = 255
        preview = propose(result_for(mask), self.settings)
        self.assertIsNone(preview.target_instance_id)
        self.assertEqual(preview.proposals[0].status, "rejected")

    def test_two_small_separate_objects_cannot_be_bridged_by_one_proposal(self):
        background = np.zeros((300, 300, 3), np.uint8)
        image = background.copy()
        cv2.circle(image, (100, 150), 24, (220, 220, 220), -1)
        cv2.circle(image, (190, 150), 24, (220, 220, 220), -1)
        result = segment(image, background, Settings(min_area=100))
        self.assertEqual(len(result.instances), 2)
        preview = propose(result, self.settings)
        self.assertIsNone(preview.target_instance_id)
        self.assertTrue(all(item.status == "rejected" for item in preview.proposals))

    def test_clipped_mask_is_rejected(self):
        mask = np.zeros((300, 300), np.uint8)
        mask[20:280, 20:280] = 255
        self.assertIsNone(propose(result_for(mask, clipped=True), self.settings).target_instance_id)

    def test_translation_changes_point_but_preserves_axis(self):
        first = np.zeros((400, 400), np.uint8)
        first[70:155, 40:270] = 255
        second = np.zeros_like(first)
        second[120:205, 70:300] = 255
        a = propose(result_for(first), self.settings).proposals[0]
        b = propose(result_for(second), self.settings).proposals[0]
        np.testing.assert_allclose(np.subtract(b.midpoint_xy_px, a.midpoint_xy_px), (30, 50))
        self.assertLess(axis_error(a.cup_axis_angle_deg, b.cup_axis_angle_deg), 0.001)

    def test_saved_preview_explicitly_has_no_robot_pose_or_calibrated_scale(self):
        color, background, _, _, _ = make_scene()
        result = segment(color, background, Settings())
        preview = propose(result, self.settings)
        with tempfile.TemporaryDirectory() as root:
            path = save_run(root, color, background, result, Settings(), grasp_preview=preview)
            saved = json.loads((path / "grasp_preview.json").read_text())
            self.assertIsNone(saved["robot_pose"])
            self.assertFalse(saved["scale_calibrated"])
            self.assertTrue((path / "grasp_overlay.png").exists())
            self.assertIsNotNone(saved["target_instance_id"])
            for item in saved["proposals"]:
                if item["status"] == "candidate_2d":
                    self.assertGreaterEqual(item["cup_axis_angle_deg"], 0)
                    self.assertLess(item["cup_axis_angle_deg"], 180)

    def test_demo_cli_can_export_grasp_preview(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with redirect_stdout(io.StringIO()):
                status = main(["demo", "--mode", "rgb", "--grasp-preview", "--data-dir",
                               str(root / "data"), "--output", str(root / "results")])
            self.assertEqual(status, 0)
            self.assertEqual(len(list((root / "results").glob("*/grasp_preview.json"))), 1)


if __name__ == "__main__":
    unittest.main()
