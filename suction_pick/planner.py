"""Multi-cup suction grasp planner after Jiang et al., arXiv:2304.10693 (Sec. 5).

Steps (paper section in brackets):
 1. Voxelise the graspable (affordable) points, voxel l = 5 mm; a voxel is occupied when it
    holds enough points. [5.3]
 2. Gripper orientations: histogram the affordable surface normals into delta-alpha = 5 deg
    (azimuth theta, polar phi) bins, keep the most frequent top 10 %, and combine each with
    roll gamma in delta-alpha steps: O = Rz(theta) Ry'(phi) Rz''(gamma). [5.4, Eq. 1, 4-6]
    (The paper also widens each normal bin to the gripper-axis bins within epsilon_1 via a
    look-up table; here the bin direction itself is used. Step 4 applies the same
    epsilon_1 test to every cup.)
 3. For every orientation, the cup centres (rotated into the TCP frame and expressed in
    voxels) form the encoded kernel; the paper writes cup i into the kernel as 10^-i,
    convolves it with the voxel grid and reads cup i's activation back from decimal digit i.
    The kernel is non-zero at exactly one voxel per cup, so that convolution equals
    "voxel grid shifted by cup i's offset" for each cup - computed that way here (exact, no
    floating-point digit decoding). TCP positions where >= 2 cups (or all cups for a shared
    vacuum line) land on occupied voxels are candidates. [5.5-5.7, Eq. 7, Alg. 3-4]
 4. Per active cup, the nearest affordable point must be within epsilon_2 and its normal
    within epsilon_1 = 11.5 deg of the gripper axis. [5.7, Eq. 2-3]
 5. Rank: most objects grasped first (maxObj = distinct regions under the cups; limited by
    PlannerParams.max_objects, default 1 = one parcel per pick, the paper has no limit), then
    J = J_center + J_var + J_orient. [5.8, Eq. 8-10, Alg. 5] The paper leaves J_center's
    formula and Poly_orient undefined; here, per grasped object, J_center is the distance
    from the mean of its cups to the object's centre, J_var the spread of those distances
    over objects, and J_orient rewards cups lined up with an elongated object's long axis.
    Lower distances are better: score = -(J_center + sqrt(J_var)) + w * J_orient.
 6. Added (not in the paper): (0) collision model of the whole end effector - cup footprint,
    tool body + flange + wrist and the wrist camera, each a column along the approach axis -
    so bin walls / neighbours near the path reject a grasp; (a) align the approach axis with the measured surface normal
    under the cups (removes the 5 deg sampling error on parcels tilted in pitch AND roll);
    (b) every active cup must touch its surface (gap <= max_seal_gap); (c) reject grasps
    where scene points stick up into the gripper's footprint - the gripper body would hit them.
 7. Fallback when no multi-cup grasp exists and cups are independent: single cup at the
    best affordable point (paper Alg. 1 line 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .affordance import SurfaceMap
from .config import Gripper, PlannerParams
from .geometry import angle_deg


@dataclass
class Grasp:
    tcp: np.ndarray            # (3,) planning frame, mm, on the surface
    R: np.ndarray              # (3, 3) gripper frame: col 2 = approach normal (out of surface)
    cups: np.ndarray           # (N, 3) cup centres, planning frame
    active: np.ndarray         # (N,) bool
    cup_labels: np.ndarray     # (N,) region id under each cup (0 = none)
    n_objects: int
    score: float
    terms: dict = field(default_factory=dict)

    @property
    def R_tool(self) -> np.ndarray:
        """Robot tool orientation: tool Z points INTO the surface (gripper frame turned 180 deg about X)."""
        return self.R @ np.diag([1.0, -1.0, -1.0])

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.R[2, 2], -1, 1))))

    def as_dict(self) -> dict:
        return {
            "tcp_mm": np.round(self.tcp, 2).tolist(),
            "approach_normal": np.round(self.R[:, 2], 4).tolist(),
            "R_gripper": np.round(self.R, 5).tolist(),
            "R_tool": np.round(self.R_tool, 5).tolist(),
            "cups_mm": np.round(self.cups, 2).tolist(),
            "active_cups": self.active.astype(int).tolist(),
            "cup_regions": self.cup_labels.astype(int).tolist(),
            "n_objects": self.n_objects, "score": round(self.score, 3),
            "tilt_deg": round(self.tilt_deg, 2),
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.terms.items()},
        }


@dataclass
class Regions:
    centroid: dict            # label -> (3,)
    major_axis: dict          # label -> (3,) unit
    elongation: dict          # label -> [0, 1]


def region_stats(surface: SurfaceMap) -> Regions:
    centroid, axis, elong = {}, {}, {}
    for lab in np.unique(surface.labels[surface.labels > 0]):
        pts = surface.points[surface.labels == lab]
        c = pts.mean(axis=0)
        vals, vecs = np.linalg.eigh(np.cov((pts - c).T) + 1e-9 * np.eye(3))
        centroid[int(lab)], axis[int(lab)] = c, vecs[:, 2]
        elong[int(lab)] = float(1.0 - np.sqrt(max(vals[1], 0) / max(vals[2], 1e-9)))
    return Regions(centroid, axis, elong)


def sample_orientations(normals: np.ndarray, p: PlannerParams, labels=None) -> list[np.ndarray]:
    """Paper Sec. 5.4: frequent normal bins x roll samples -> list of 3x3 rotations.

    Added: each region's own most frequent bin is always kept too, otherwise one tilted
    object among many flat ones never makes the global top 10 %.
    """
    step = np.radians(p.angle_step)
    theta = np.arctan2(normals[:, 1], normals[:, 0])
    phi = np.arccos(np.clip(normals[:, 2], -1, 1))
    ii = np.round((theta + np.pi) / step).astype(int)
    jj = np.round(phi / step).astype(int)
    ii[jj == 0] = 0  # azimuth is meaningless for a vertical normal
    bins, counts = np.unique(np.column_stack([ii, jj]), axis=0, return_counts=True)
    order = np.argsort(-counts)
    keep = {tuple(b) for b in bins[order[:max(1, int(np.ceil(p.top_normal_fraction * len(bins))))]]}
    if labels is not None:
        for lab in np.unique(labels):
            b, c = np.unique(np.column_stack([ii, jj])[labels == lab], axis=0, return_counts=True)
            keep.add(tuple(b[np.argmax(c)]))
    rolls = np.radians(np.arange(0.0, p.roll_range, p.angle_step))
    out = []
    for i, j in keep:
        th, ph = i * step - np.pi, j * step
        if np.degrees(ph) > p.max_tilt:
            continue
        for g in rolls:
            out.append(Rotation.from_euler("ZYZ", [th, ph, g]).as_matrix())
    return out


def plan(surface: SurfaceMap, gripper: Gripper, p: PlannerParams, rng=None,
         obstacles=None) -> tuple[list[Grasp], dict]:
    """obstacles: extra (N, 3) points the end effector must not touch (e.g. virtual bin walls)."""
    rng = rng or np.random.default_rng(0)
    info = {}
    aff = surface.score > 0
    if not aff.any():
        return [], {"reason": "no graspable surface found"}
    pts, nrm, lab = surface.points[aff], surface.normals[aff], surface.labels[aff]
    regions = region_stats(surface)
    cups_g = gripper.cups_gripper_frame()
    need = gripper.min_active()

    # 1. voxel grid of affordable points (+ region label per voxel)
    l = p.voxel
    bmin = pts.min(axis=0)
    idx = np.floor((pts - bmin) / l).astype(int)
    shape = idx.max(axis=0) + 1
    if np.prod(shape) > p.max_grid_voxels:
        ext = (shape * l / 1000).round(2).tolist()
        return [], {"reason": f"graspable area spans {ext} m - far more than a bin. Lower "
                              "--max_range or check what the camera sees (overlay.png)."}
    reach = int(np.ceil(np.linalg.norm(cups_g, axis=1).max() / l)) + 1  # max cup offset, voxels
    pad = reach
    counts = np.zeros(shape + 2 * pad, np.int32)
    np.add.at(counts, tuple((idx + pad).T), 1)
    V = counts >= p.min_points_per_voxel
    L = np.zeros(counts.shape, np.int32)
    L[tuple((idx + pad).T)] = lab
    info["voxels"] = int(V.sum())
    info["grid"] = list(map(int, V.shape))

    # 2. orientations
    sel = slice(None) if len(nrm) < 50000 else rng.choice(len(nrm), 50000, replace=False)
    orients = sample_orientations(nrm[sel], p, lab[sel])
    info["orientations"] = len(orients)

    # 3. encoded-kernel convolution, one shift per cup; coarse ranking vectorised per
    #    orientation, keeping only its best `keep` candidates (a big flat object can give
    #    hundreds of thousands of TCP positions).
    max_lab = int(surface.labels.max())
    cent = np.zeros((max_lab + 1, 3))
    for lb, c in regions.centroid.items():
        cent[lb] = c
    keep = max(50, 4 * p.max_detailed // max(1, len(orients)))
    cand, raw, multi = [], 0, 0  # cand: (n_obj, J_center, orientation idx, voxel idx (3,), active (N,))
    for k, R in enumerate(orients):
        offs = np.round((cups_g @ R.T) / l).astype(int)          # cup offsets from TCP, voxels
        act = np.stack([np.roll(V, tuple(-o), axis=(0, 1, 2)) for o in offs])  # act[i][x] = V[x + o_i]
        hits = act.sum(axis=0) >= need
        hits[:pad], hits[-pad:], hits[:, :pad], hits[:, -pad:] = False, False, False, False
        hits[:, :, :pad], hits[:, :, -pad:] = False, False
        X = np.argwhere(hits)
        if len(X) == 0:
            continue
        raw += len(X)
        A = np.stack([act[i][tuple(X.T)] for i in range(len(offs))], axis=1)        # (M, N)
        labs = np.stack([L[tuple((X + o).T)] for o in offs], axis=1) * A          # (M, N)
        tcp = bmin + (X - pad + 0.5) * l
        cups = tcp[:, None, :] + (cups_g @ R.T)[None]                              # (M, N, 3)
        n_obj, j_center = _coarse_terms(cups, labs, cent)
        allowed = np.flatnonzero(n_obj <= p.max_objects)   # e.g. 1 = all cups on one parcel
        multi += len(n_obj) - len(allowed)
        best = allowed[np.lexsort((j_center[allowed], -n_obj[allowed]))][:keep]
        cand += [(int(n_obj[b]), float(j_center[b]), k, X[b], A[b]) for b in best]
    info["raw_candidates"] = raw
    info["dropped_multi_object"] = multi
    if not cand:
        return _fallback(surface, gripper, p, regions, info, obstacles)
    cand.sort(key=lambda c: (-c[0], c[1]))
    order = range(len(cand))
    cand = [(k, x, a) for _, _, k, x, a in cand]

    # 4-6. detailed checks on the best candidates
    tree = cKDTree(pts)
    scene = surface.points[np.isfinite(surface.points).all(axis=2)]
    scene = scene[rng.choice(len(scene), min(len(scene), 60000), replace=False)]
    if obstacles is not None and len(obstacles):
        scene = np.vstack([scene, obstacles])
    ee = ee_model(gripper, p.hover_radius)
    grasps, rejected = [], {"normal": 0, "contact": 0, "seal": 0, "collision": 0, "multi_object": 0}
    for i in order[:p.max_detailed]:
        k, x, act = cand[i]
        g = _refine(orients[k], bmin + (x - pad + 0.5) * l, act, cups_g, tree, pts, nrm, lab,
                    scene, ee, regions, p, rejected)
        if g is not None:
            grasps.append(g)
    info["rejected"] = rejected
    grasps = _nms(sorted(grasps, key=lambda g: (-g.n_objects, -g.score)), p)
    if not grasps:
        return _fallback(surface, gripper, p, regions, info, obstacles)
    return grasps[:p.max_results], info


def _coarse_terms(cups, labs, cent):
    """Vectorised (n_objects, J_center) for M candidates: cups (M,N,3), labs (M,N) (0 = inactive)."""
    on = labs > 0
    same = (labs[:, :, None] == labs[:, None, :]) & on[:, :, None] & on[:, None, :]   # (M,N,N)
    size = same.sum(axis=2)                                                           # group size per cup
    gmean = (same[..., None] * cups[:, None, :, :]).sum(axis=2) / np.maximum(size, 1)[..., None]
    d = np.linalg.norm(gmean - cent[labs], axis=2)                                   # per cup
    s = np.sort(labs, axis=1)
    n_obj = ((s > 0) & np.concatenate([np.ones((len(s), 1), bool), s[:, 1:] != s[:, :-1]], axis=1)).sum(axis=1)
    j_center = np.where(on, d / np.maximum(size, 1), 0).sum(axis=1) / np.maximum(n_obj, 1)
    return n_obj, np.where(n_obj > 0, j_center, 1e9)


def _terms(cups, act, labs, R, regions: Regions):
    groups = {}
    for c, a, lb in zip(cups, act, labs):
        if a and lb > 0:
            groups.setdefault(int(lb), []).append(c)
    if not groups:
        return 0, 1e9, 0.0, 0.0
    d = np.array([np.linalg.norm(np.mean(g, axis=0) - regions.centroid[lb]) for lb, g in groups.items()])
    orient = []
    for lb, g in groups.items():
        if len(g) >= 2:
            axis = g[-1] - g[0]
            axis /= np.linalg.norm(axis)
            orient.append(abs(axis @ regions.major_axis[lb]) * regions.elongation[lb])
    return len(groups), float(d.mean()), float(d.var()), float(np.mean(orient)) if orient else 0.0


def ee_model(gripper, hover_radius=0.0):
    """Collision model: cup footprint box + upward columns (tool body/wrist, camera)."""
    cups_g = gripper.cups_gripper_frame()
    half = np.abs(cups_g[:, :2]).max(axis=0) + gripper.cup_radius + gripper.body_margin
    cols = gripper.columns()
    reach = max([float(np.hypot(*half))] + [float(np.hypot(cx, cy) + r) for cx, cy, r, _ in cols])
    # Hover height looks further out than the gripper itself, so the joint move to the hover
    # point ends above the bin rim even when the parcel is far from the walls.
    return {"half": half, "columns": cols, "clear_radius": max(reach + 20.0, hover_radius)}


def ee_hit_mask(q, ee, p):
    """Which scene points (gripper frame, (N, 3)) are inside the end effector or above it."""
    hit = (np.abs(q[:, 0]) < ee["half"][0]) & (np.abs(q[:, 1]) < ee["half"][1]) & (q[:, 2] > p.collision_tol)
    for cx, cy, r, z0 in ee["columns"]:
        hit |= ((q[:, 0] - cx) ** 2 + (q[:, 1] - cy) ** 2 < r * r) & (q[:, 2] > z0)
    return hit


def ee_collisions(q, ee, p):
    """Number of scene points (gripper frame, (N, 3)) inside the end effector or above it."""
    return int(ee_hit_mask(q, ee, p).sum())


def clear_height(q, ee):
    """Highest scene point (along the approach axis) near the approach line, e.g. a bin rim."""
    near = q[:, 0] ** 2 + q[:, 1] ** 2 < ee["clear_radius"] ** 2
    return float(max(0.0, q[near, 2].max())) if near.any() else 0.0


def _refine(R, tcp, act, cups_g, tree, pts, nrm, lab, scene, ee, regions, p, rejected):
    active = np.flatnonzero(act)

    def contacts(R, cups):
        """Nearest affordable point per active cup, with the paper's Eq. 3 / Eq. 2 checks."""
        out = {}
        for i in active:
            dist, j = tree.query(cups[i])
            if dist > p.contact_tol:                    # Eq. 3
                rejected["contact"] += 1
                return None
            if angle_deg(nrm[j], R[:, 2]) > p.normal_tol:   # Eq. 2
                rejected["normal"] += 1
                return None
            out[i] = j
        return out

    idx = contacts(R, tcp + cups_g @ R.T)
    if idx is None:
        return None
    if p.refine_axis:
        # Added: the sampled axis is quantised to angle_step (5 deg), so a parcel tilted in
        # both pitch and roll is approached up to ~2.5 deg crooked. Align the approach axis
        # with the measured surface normal under the cups (mean of their patch normals),
        # keep the cup direction (projected into the new plane), and check again.
        z = np.mean([nrm[j] for j in idx.values()], axis=0)
        z /= np.linalg.norm(z)
        x = R[:, 0] - (R[:, 0] @ z) * z
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(z, x), z])
        idx = contacts(R, tcp + cups_g @ R.T)
        if idx is None:
            return None
    z = R[:, 2]
    cups = tcp + cups_g @ R.T
    gaps = np.array([(pts[j] - cups[i]) @ z for i, j in idx.items()])
    tcp = tcp + gaps.mean() * z                          # put the cup plane on the surface
    if np.abs(gaps - gaps.mean()).max() > p.max_seal_gap:
        rejected["seal"] += 1                            # added: every active cup must touch
        return None
    cups = tcp + cups_g @ R.T
    labs = np.zeros(len(cups), int)
    for i, j in idx.items():
        labs[i] = lab[j]
    q = (scene - tcp) @ R                                # scene in the gripper frame
    if ee_collisions(q, ee, p) > p.collision_max_points:  # cups, tool body, wrist, camera
        rejected["collision"] += 1
        return None
    # The same grasp turned 180 deg (robot may use it for less wrist motion) puts the camera
    # on the other side: record whether that is clear too.
    turned_clear = ee_collisions(q * np.array([-1.0, -1.0, 1.0]), ee, p) <= p.collision_max_points
    n_obj, j_center, j_var, j_orient = _terms(cups, act, labs, R, regions)
    if n_obj > p.max_objects:                            # cups on more than one parcel
        rejected["multi_object"] += 1
        return None
    score = -(j_center + np.sqrt(j_var)) + p.orient_weight * j_orient
    return Grasp(tcp, R, cups, np.asarray(act, bool), labs, n_obj, float(score),
                 {"J_center_mm": j_center, "J_var_mm2": j_var, "J_orient": j_orient,
                  "seal_gap_mm": float(np.abs(gaps - gaps.mean()).max()), "method": "multi-cup",
                  "clear_height_mm": clear_height(q, ee), "turned_clear": bool(turned_clear)})


def _nms(grasps, p, per_object_set=2):
    """Drop near-duplicates and keep at most `per_object_set` grasps per set of objects,
    so the list also shows the best grasp for every other object (input is sorted best-first)."""
    kept, per_set = [], {}
    for g in grasps:
        key = tuple(sorted(set(g.cup_labels[g.active].tolist())))
        if per_set.get(key, 0) >= per_object_set:
            continue
        if all(np.linalg.norm(g.tcp - k.tcp) > p.nms_distance or
               abs(g.R[:, 0] @ k.R[:, 0]) < np.cos(np.radians(10)) for k in kept):
            kept.append(g)
            per_set[key] = per_set.get(key, 0) + 1
    return kept


def _fallback(surface, gripper, p, regions, info, obstacles=None):
    """Single-cup grasp at the best affordable point (only with independent cups)."""
    if not gripper.independent_cups:
        info["reason"] = ("no pose puts all cups on graspable surface (shared vacuum line: every "
                          "cup must seal). Objects may be smaller than the cup layout.")
        return [], info
    info["fallback"] = "single-cup"
    score = surface.score.copy()
    cups_g = gripper.cups_gripper_frame()
    ee = ee_model(gripper, p.hover_radius)
    scene = surface.points[np.isfinite(surface.points).all(axis=2)]
    scene = scene[np.random.default_rng(0).choice(len(scene), min(len(scene), 60000), replace=False)]
    if obstacles is not None and len(obstacles):
        scene = np.vstack([scene, obstacles])
    for flat in np.argsort(-score, axis=None)[:200]:
        v, u = np.unravel_index(flat, score.shape)
        if score[v, u] <= 0:
            break
        point, n = surface.points[v, u], surface.normals[v, u]
        R0 = Rotation.align_vectors([n], [[0, 0, 1]])[0].as_matrix()
        for g in np.radians(np.arange(0.0, 360.0, p.angle_step * 3)):
            R = R0 @ Rotation.from_euler("z", g).as_matrix()
            for ci in range(len(cups_g)):
                tcp = point - R @ cups_g[ci]
                q = (scene - tcp) @ R
                if ee_collisions(q, ee, p) <= p.collision_max_points:
                    act = np.zeros(len(cups_g), bool)
                    act[ci] = True
                    labs = np.zeros(len(cups_g), int)
                    labs[ci] = surface.labels[v, u]
                    return [Grasp(tcp, R, tcp + cups_g @ R.T, act, labs, 1, float(score[v, u]),
                                  {"method": "single-cup fallback", "clear_height_mm": clear_height(q, ee),
                                   "turned_clear": ee_collisions(q * np.array([-1.0, -1.0, 1.0]), ee, p)
                                   <= p.collision_max_points})], info
    info["reason"] = "no collision-free single-cup grasp"
    return [], info
