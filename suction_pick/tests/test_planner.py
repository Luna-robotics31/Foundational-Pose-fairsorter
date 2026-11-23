"""Offline checks (no camera, no robot):  python -m unittest discover -s suction_pick/tests -t . -v"""
import unittest

import numpy as np

from suction_pick import synthetic as S
from suction_pick.affordance import compute_affordance
from suction_pick.config import AffordanceParams, Gripper, PlannerParams
from suction_pick.geometry import T_UP_FROM_CAMERA, depth_to_points, transform
from suction_pick.planner import plan


def run(boxes=None, gripper=None, params=None):
    depth, _ = S.render(boxes=boxes)
    gripper = gripper or Gripper()
    pts = transform(T_UP_FROM_CAMERA, depth_to_points(depth, S.K))
    surf = compute_affordance(pts, depth, S.K[0, 0], gripper.cup_radius, AffordanceParams())
    return plan(surf, gripper, params or PlannerParams())


def on(g, boxes=None):
    return sorted(S.box_under(c[:2], boxes=boxes) for c, a in zip(g.cups, g.active) if a)


class Planner(unittest.TestCase):
    def test_reference_scene_multi_object(self):
        grasps, _ = run(params=PlannerParams(max_objects=99))   # the paper's mode
        self.assertEqual(on(grasps[0]), ["small_A", "small_B"])        # 2 objects, one grasp
        self.assertLess(np.linalg.norm(grasps[0].tcp[:2] - [0, 60]), 5)
        self.assertAlmostEqual(grasps[0].tcp[2], -520, delta=3)         # on the box tops
        self.assertTrue(any(on(g) == ["large_C", "large_C"] for g in grasps))
        tilted = [g for g in grasps if on(g) == ["tilted_D", "tilted_D"]]
        self.assertTrue(tilted and abs(tilted[0].tilt_deg - 20) < 3)
        self.assertFalse(any("tiny_E" in on(g) for g in grasps))
        for g in grasps:                                                 # tool Z into the surface
            self.assertAlmostEqual(g.R_tool[:, 2] @ g.R[:, 2], -1.0, places=6)

    def test_one_parcel_per_grasp_by_default(self):
        grasps, info = run()
        self.assertGreater(info["dropped_multi_object"], 0)
        self.assertTrue(grasps)
        for g in grasps:
            self.assertEqual(g.n_objects, 1)
            self.assertEqual(len(set(on(g))), 1)
        self.assertFalse(any(set(on(g)) & {"small_A", "small_B"} for g in grasps))

    def test_cups_parallel_to_surface(self):
        for g in run()[0]:
            spread = np.ptp((g.cups - g.tcp) @ g.R[:, 2])
            self.assertLess(spread, 1e-6)                                # cups in one plane

    def test_collision_with_tall_neighbour(self):
        boxes = [("big", (0.0, 0.0), (160.0, 100.0), 40.0, 0.0),
                 ("post", (0.0, 64.0), (40.0, 20.0), 160.0, 0.0)]          # tall post at the edge
        free, info_free = run(boxes, params=PlannerParams(collision_tol=1e6))
        safe, info = run(boxes)
        self.assertGreater(info["rejected"]["collision"], 0)
        post_top = S.TABLE_Z + 160
        for g in safe:                                                   # no post inside the footprint
            corners = np.array([[sx, sy, 0] for sx in (-55, 55) for sy in (-20, 20)])
            xy = (g.tcp + corners @ g.R.T)[:, :2]
            post = (abs(xy[:, 0]).min() < 20 or (xy[:, 0].min() < 0 < xy[:, 0].max())) and xy[:, 1].max() > 54
            self.assertFalse(post and post_top > g.tcp[2], f"gripper over the post: {xy}")
        self.assertTrue(free and safe)

    def test_parcel_tilted_in_pitch_and_roll(self):
        from cube_pick.robot.pick_cube import pose_to_T
        from suction_pick.geometry import angle_deg
        from suction_pick.robot import tool_pose
        for tilt in [(15, 10), (-20, 20), (25, -20), (35, 25)]:
            boxes = [("parcel", (0.0, 0.0), (160.0, 110.0), 80.0, tilt)]
            grasps, _ = run(boxes)
            n = S.true_normal(tilt)
            g = grasps[0]
            self.assertLess(angle_deg(g.R[:, 2], n), 0.5, tilt)            # approach = surface normal
            tool_z = pose_to_T(tool_pose(g.tcp, g.R_tool))[:3, 2]           # Fairino rx,ry,rz convention
            self.assertLess(angle_deg(tool_z, -n), 0.5, tilt)               # tool Z points into the parcel
            self.assertLess(g.terms["seal_gap_mm"], 1.0, tilt)              # both cups touch
            self.assertEqual(on(g, boxes), ["parcel", "parcel"])
        too_steep, _ = run([("parcel", (0.0, 0.0), (160.0, 110.0), 80.0, (40, 30))])  # 45.5 deg
        self.assertEqual(too_steep, [])

    def test_object_tilt_report(self):
        from suction_pick.affordance import object_tilts
        boxes = [("level", (-130.0, 70.0), (90.0, 90.0), 60.0, 0.0),
                 ("p15_r10", (-70.0, -80.0), (110.0, 90.0), 70.0, (15, 10)),
                 ("p-8_r25", (90.0, -80.0), (110.0, 90.0), 70.0, (-8, 25))]
        depth, _ = S.render(boxes=boxes)
        pts = transform(T_UP_FROM_CAMERA, depth_to_points(depth, S.K))
        surf = compute_affordance(pts, depth, S.K[0, 0], 15, AffordanceParams())
        found = {S.box_under(t["centre_mm"][:2], boxes=boxes): t for t in object_tilts(surf)}
        self.assertFalse(found["level"]["tilted"])
        for name, (pitch, roll) in (("p15_r10", (15, 10)), ("p-8_r25", (-8, 25))):
            self.assertTrue(found[name]["tilted"])
            self.assertAlmostEqual(found[name]["pitch_deg"], pitch, delta=0.5)
            self.assertAlmostEqual(found[name]["roll_deg"], roll, delta=0.5)

    def test_small_objects_shared_vs_independent(self):
        boxes = [("p", (-60.0, 0.0), (36.0, 36.0), 50.0, 0.0), ("q", (90.0, 40.0), (36.0, 36.0), 70.0, 0.0)]
        grasps, info = run(boxes)                                        # shared vacuum: impossible
        self.assertEqual(grasps, [])
        self.assertIn("shared vacuum", info["reason"])
        grasps, info = run(boxes, gripper=Gripper(independent_cups=True))
        self.assertEqual(info.get("fallback"), "single-cup")
        self.assertEqual(int(grasps[0].active.sum()), 1)
        self.assertIn(on(grasps[0], boxes)[0], ("p", "q"))


if __name__ == "__main__":
    unittest.main()
