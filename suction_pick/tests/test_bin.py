"""Bin selection (python run.py bin) on a synthetic scene: floor, table, two parcels on the table."""
import unittest

import cv2
import numpy as np

from suction_pick.app.bin_ui import BinPicker
from suction_pick.vision.geometry import depth_to_points

K = np.array([[520.0, 0, 320], [0, 520.0, 240], [0, 0, 1]])
T = np.eye(4)
T[:3, :3] = np.diag([1, -1, -1])          # camera looking straight down from z = 1500 mm
T[:3, 3] = [0, 0, 1500]


def scene():
    depth = np.full((480, 640), 1400.0)   # floor, z 100
    depth[90:400, 120:520] = 1000.0       # table, z 500, fully in view
    depth[200:300, 250:330] = 800.0       # parcels on the table
    depth[150:230, 380:470] = 850.0
    depth += np.random.default_rng(0).normal(0, 0.5, depth.shape)
    bgr = np.full((480, 640, 3), 90, np.uint8)
    return BinPicker(bgr, depth_to_points(depth.astype(np.float32), K), K, T)


class BinSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = scene()

    def setUp(self):
        self.ui.handle_key(ord("1") + self.ui.auto)          # back to the automatic choice

    def test_auto_is_table_not_floor_or_parcel(self):
        b = self.ui.board()
        self.assertAlmostEqual(b["corners_base"][:, 2].mean(), 500, delta=3)
        self.assertAlmostEqual(b["size_mm"][0], 400 * 1000 / 520, delta=15)
        self.assertAlmostEqual(b["floor_z"], 100, delta=3)
        self.assertEqual(b["cut_sides"], [])

    def test_number_key_selects_other_surface(self):
        other = 1 - self.ui.auto
        self.ui.handle_key(ord("1") + other)
        self.assertEqual(self.ui.sel, other)
        self.assertAlmostEqual(self.ui.board()["corners_base"][:, 2].mean(), 100, delta=3)

    def test_manual_four_clicks(self):
        ui = self.ui
        ui.handle_key(ord("m"))
        self.assertIsNone(ui.handle_key(13))                  # Enter before 4 clicks: not saved
        for x, y in [(520, 400), (120, 90), (120, 400), (520, 90)]:   # any order
            ui.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y)
        self.assertEqual(ui.handle_key(13), "save")
        b = ui.board()
        self.assertAlmostEqual(b["corners_base"][:, 2].mean(), 500, delta=3)   # parcels ignored
        self.assertAlmostEqual(b["size_mm"][0], 400 * 1000 / 520, delta=5)
        self.assertAlmostEqual(b["size_mm"][1], 310 * 1000 / 520, delta=5)
        ui.handle_key(ord("r"))
        self.assertEqual(ui.clicks, [])
        self.assertEqual(ui.handle_key(ord("q")), "quit")


if __name__ == "__main__":
    unittest.main()
