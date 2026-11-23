"""Behavior checks against known masks, depth failures, and saved artifacts."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from parcel_lab.artifacts import read_depth, save_run, write_image
from parcel_lab.cli import main
from parcel_lab.demo import make_scene
from parcel_lab.segmentation import Settings, segment
from parcel_lab.sources import Frame, median_reference


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.color, self.background, self.depth, self.reference_depth, self.truth = make_scene()
        self.settings = Settings(min_area=500)

    def test_three_separated_parcels_have_correct_individual_masks(self):
        for mode in ("rgb", "depth", "hybrid"):
            with self.subTest(mode=mode):
                result = segment(self.color, self.background, replace(self.settings, mode=mode), self.depth, self.reference_depth)
                self.assertEqual(len(result.instances), 3)
                matched = set()
                for truth_id in (1, 2, 3):
                    actual = self.truth == truth_id
                    scores = []
                    for candidate in result.instances:
                        predicted = candidate.mask > 0
                        scores.append(np.count_nonzero(actual & predicted) / np.count_nonzero(actual | predicted))
                    best = int(np.argmax(scores))
                    self.assertGreater(scores[best], 0.97)
                    self.assertNotIn(best, matched)
                    matched.add(best)

    def test_empty_bin_has_no_instances(self):
        for mode in ("rgb", "depth", "hybrid"):
            result = segment(self.background, self.background, replace(self.settings, mode=mode), self.reference_depth, self.reference_depth)
            self.assertEqual(result.instances, [])

    def test_depth_finds_a_parcel_same_color_as_background(self):
        result = segment(self.background, self.background, replace(self.settings, mode="depth"), self.depth, self.reference_depth)
        self.assertEqual(len(result.instances), 3)

    def test_valid_depth_rejects_color_only_shadow_in_hybrid(self):
        changed = self.background.copy()
        changed[80:220, 60:230] = 0
        result = segment(changed, self.background, replace(self.settings, mode="hybrid"), self.reference_depth, self.reference_depth)
        self.assertEqual(len(result.instances), 0)

    def test_zero_nan_and_farther_depth_are_not_foreground(self):
        depth = self.reference_depth.copy()
        depth[50:120, 50:120] = 0
        depth[150:220, 150:220] = np.nan
        depth[260:330, 260:330] = 1200
        result = segment(self.background, self.background, replace(self.settings, mode="depth"), depth, self.reference_depth)
        self.assertEqual(len(result.instances), 0)

    def test_hybrid_uses_rgb_only_where_depth_is_missing(self):
        depth = np.zeros_like(self.depth)
        result = segment(self.color, self.background, replace(self.settings, mode="hybrid"), depth, self.reference_depth)
        self.assertEqual(len(result.instances), 3)
        self.assertTrue(result.warnings)
        self.assertTrue(all(item.median_depth_mm is None for item in result.instances))

    def test_roi_excludes_other_parcels_and_flags_clipped_object(self):
        result = segment(self.color, self.background, replace(self.settings, roi=(80, 90, 180, 150)))
        self.assertEqual(len(result.instances), 1)
        self.assertTrue(result.instances[0].touches_roi_boundary)
        self.assertTrue(np.all(result.foreground[:90] == 0))

    def test_small_debris_is_filtered_without_merging_separated_parcels(self):
        color = self.background.copy()
        color[60:180, 60:180] = 230
        color[60:180, 186:306] = 230
        color[300:304, 300:304] = 230
        result = segment(color, self.background, self.settings)
        self.assertEqual(len(result.instances), 2)

    def test_touching_parcels_are_one_component_not_falsely_claimed_separate(self):
        color = self.background.copy()
        color[60:180, 60:180] = 230
        color[60:180, 180:300] = 230
        self.assertEqual(len(segment(color, self.background, self.settings).instances), 1)

    def test_mismatched_and_unpaired_depth_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "both"):
            segment(self.color, self.background, self.settings, self.depth)
        with self.assertRaisesRegex(ValueError, "aligned"):
            segment(self.color, self.background, self.settings, self.depth[::2], self.reference_depth)
        with self.assertRaisesRegex(ValueError, "identical"):
            segment(self.color, self.background[::2], self.settings)
        with self.assertRaisesRegex(ValueError, "requires"):
            segment(self.color, self.background, replace(self.settings, mode="depth"))

    def test_saved_masks_labels_crops_and_report_agree(self):
        result = segment(self.color, self.background, self.settings, self.depth, self.reference_depth)
        with tempfile.TemporaryDirectory() as root:
            path = save_run(root, self.color, self.background, result, self.settings, self.depth, self.reference_depth)
            report = json.loads((path / "instances.json").read_text())
            self.assertEqual(report["count"], 3)
            labels = cv2.imread(str(path / "labels.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(labels.dtype, np.uint16)
            for record in report["instances"]:
                mask = cv2.imread(str(path / record["mask"]), cv2.IMREAD_GRAYSCALE)
                np.testing.assert_array_equal(mask > 0, labels == record["id"])
                crop = cv2.imread(str(path / record["crop"]), cv2.IMREAD_UNCHANGED)
                x, y, w, h = record["bbox_xywh_px"]
                np.testing.assert_array_equal(crop[:, :, 3], mask[y:y + h, x:x + w])
            np.testing.assert_array_equal(read_depth(path / "depth_mm.npy"), self.depth)
            second = save_run(root, self.color, self.background, result, self.settings)
            self.assertNotEqual(path, second)

    def test_expected_count_mismatch_has_nonzero_exit_but_preserves_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_image(root / "scene.png", self.color)
            write_image(root / "empty.png", self.background)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(["image", "--image", str(root / "scene.png"), "--background",
                               str(root / "empty.png"), "--expected-count", "4", "--output", str(root / "results")])
            self.assertEqual(status, 1)
            reports = list((root / "results").glob("*/instances.json"))
            self.assertEqual(len(reports), 1)
            self.assertFalse(json.loads(reports[0].read_text())["metadata"]["count_matches"])

    def test_reference_median_ignores_invalid_depth(self):
        color = np.zeros((4, 4, 3), np.uint8)
        arrays = [np.full((4, 4), value, np.float32) for value in (0, 100, 200)]
        for array in arrays:
            array[0, 0] = np.nan
        reference = median_reference([Frame(color, array) for array in arrays])
        self.assertEqual(reference.depth_mm[1, 1], 150)
        self.assertEqual(reference.depth_mm[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
