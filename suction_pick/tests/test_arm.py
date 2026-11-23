"""FR10 model: the URDF (sim + collision shapes) must move exactly like the controller kinematics,
and the collision capsules must contain the real link meshes."""
import struct
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from suction_pick.robot import arm


def stl_vertices(path):
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    tri = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return np.unique(tri["v"].reshape(-1, 3).astype(float), axis=0) * 1000.0      # m -> mm


class FR10Model(unittest.TestCase):
    def test_urdf_flange_equals_controller_kinematics(self):
        rng = np.random.default_rng(0)
        for q in [np.zeros(6), [-42.4, -107.0, -40.1, -140.5, 97.0, 13.8]] + [rng.uniform(-170, 170, 6) for _ in range(20)]:
            T_urdf, T_dh = arm.link_frames(q)["flange"], arm.fk_frames(q)[-1]
            self.assertLess(np.linalg.norm(T_urdf[:3, 3] - T_dh[:3, 3]), 0.01)
            self.assertLess(np.abs(T_urdf[:3, :3] - T_dh[:3, :3]).max(), 1e-6)

    def test_meshes_inside_capsules(self):
        root = ET.parse(arm.URDF_FILE).getroot()
        caps = arm._capsule_table()
        for link in root.findall("link"):
            name, vis = link.get("name"), link.find("visual")
            mine = [(a, b, r) for lk, a, b, r, _ in caps if lk == name]
            if vis is None or name == "base_link":          # the base does not move: not checked
                continue
            self.assertTrue(mine, f"no capsule for {name}")
            T = arm._origin(vis.find("origin"))
            P = stl_vertices(arm.FR10_DIR / vis.find("geometry/mesh").get("filename"))
            P = P @ T[:3, :3].T + T[:3, 3]
            outside = np.min([arm.seg_distance(P, a, b) - r for a, b, r in mine], axis=0)
            self.assertLessEqual(outside.max(), 0.0, f"{name}: mesh sticks out of its capsules by {outside.max():.1f} mm")

    def test_capsules_follow_the_arm(self):
        q = [-42.4, -107.0, -40.1, -140.5, 97.0, 13.8]
        names = {c[3] for c in arm.capsules(q)}
        self.assertEqual(names, {"shoulder", "upper arm", "forearm", "wrist 1", "wrist 2", "wrist 3"})
        flange = arm.fk_frames(q)[-1][:3, 3]
        wrist3 = [c for c in arm.capsules(q) if c[3] == "wrist 3"][0]
        self.assertLess(arm.seg_distance(flange[None], wrist3[0], wrist3[1])[0], wrist3[2])   # flange inside wrist 3


if __name__ == "__main__":
    unittest.main()
