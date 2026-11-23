"""Window for `python run.py bin`: choose the bin on the camera image.

Every large horizontal surface the depth camera finds is drawn as a numbered outline. The
automatic choice (the largest surface fully in view; parcels on it are smaller, the floor runs
out of the image) is highlighted. Keys:

    Enter / Space   use the highlighted surface (or the 4 clicked corners) and save
    1 - 9           highlight another detected surface
    M               manual: click the 4 corners of the bin on the image
    R               clear the clicked corners
    Q / Esc         quit without saving
"""
from __future__ import annotations

import cv2
import numpy as np

from suction_pick.robot import workcell

WINDOW = "bin: Enter = save, 1-9 = other surface, M = click 4 corners, R = reset, Q = quit"
COLORS = [(0, 255, 255), (255, 160, 0), (255, 0, 255), (0, 200, 0), (0, 128, 255), (200, 200, 200)]


def _text(img, s, org, color=(255, 255, 255), scale=0.5):
    """Text on a dark box, readable on any background."""
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _project(K, P):
    return np.column_stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]])


class BinPicker:
    """State of the selection window (drawing and key / mouse handling, testable without a display)."""

    def __init__(self, bgr, points_cam, K, T_base_cam, max_range=2500.0):
        self.bgr, self.P, self.K, self.T = bgr, points_cam, K, T_base_cam
        self.max_range = max_range
        try:
            self.cands, self.auto = workcell.find_surfaces(points_cam, K, T_base_cam, max_range)
        except RuntimeError as exc:
            print(f"[BIN] {exc} Click the 4 corners of the bin instead.")
            self.cands, self.auto = [], None
        self.sel = self.auto
        self.manual = self.auto is None
        self.clicks: list[tuple[float, float]] = []
        self.message = ""

    # ---------------------------------------------------------------- input
    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and self.manual and len(self.clicks) < 4:
            self.clicks.append((float(x), float(y)))
            self.message = "" if len(self.clicks) < 4 else "4 corners set: Enter = save, R = redo"

    def handle_key(self, key):
        """Returns 'save', 'quit' or None (keep going)."""
        if key < 0:
            return None
        ch = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""
        if key in (27,) or ch == "q":
            return "quit"
        if ch and ch in "123456789":
            i = int(ch) - 1
            if i < len(self.cands):
                self.sel, self.manual, self.clicks, self.message = i, False, [], ""
            return None
        if ch == "m":
            self.manual, self.clicks, self.message = True, [], "click the 4 corners of the bin"
        elif ch == "r":
            self.clicks, self.message = [], "click the 4 corners of the bin" if self.manual else ""
        elif key in (13, 10, 32):
            if self.manual and len(self.clicks) < 4:
                self.message = f"click {4 - len(self.clicks)} more corner(s) first"
                return None
            if not self.manual and self.sel is None:
                self.message = "nothing detected: press M and click the 4 corners"
                return None
            return "save"
        return None

    def board(self):
        """The chosen bin, as workcell.save() expects it."""
        if self.manual:
            return workcell.board_from_pixels(self.clicks, self.P, self.K, self.T, self.cands, self.max_range)
        return workcell.board_from_surface(self.cands, self.sel, self.T)

    # ---------------------------------------------------------------- drawing
    def render(self):
        img = self.bgr.copy()
        if self.manual:
            img = (img * 0.85).astype(np.uint8)
        for i, t in enumerate(self.cands):
            corners, length, width, cut = t["rect"]
            uv = np.round(_project(self.K, corners)).astype(np.int32)
            chosen = (i == self.sel and not self.manual)
            color = COLORS[i % len(COLORS)]
            if self.manual:
                color = (110, 110, 110)
            cv2.polylines(img, [uv.reshape(-1, 1, 2)], True, color, 4 if chosen else 1, cv2.LINE_AA)
            label = (f"{i + 1}: {length:.0f} x {width:.0f} mm, z {t['z']:.0f} mm"
                     f"{', runs out of view' if cut else ''}{'  (auto)' if i == self.auto else ''}")
            H, W = img.shape[:2]
            top = uv[np.argmin(uv[:, 1])]                    # number next to the outline's top corner
            _text(img, str(i + 1), (np.clip(top[0] + 6, 4, W - 20), np.clip(top[1] + 22 + 18 * i, 95, H - 8)),
                  color, 0.7)
            _text(img, label, (10, H - 12 - 24 * (len(self.cands) - 1 - i)), color)
        if self.clicks:
            pts = np.round(self.clicks).astype(np.int32)
            for k, p in enumerate(pts):
                cv2.circle(img, tuple(int(v) for v in p), 6, (0, 0, 255), -1, cv2.LINE_AA)
                _text(img, str(k + 1), (int(p[0]) + 8, int(p[1]) - 8), (0, 0, 255), 0.6)
            if len(pts) == 4:
                c = pts.mean(axis=0)
                pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], len(pts) == 4, (0, 0, 255), 2, cv2.LINE_AA)
        if self.manual:
            head = f"MANUAL: click corner {len(self.clicks) + 1} of 4" if len(self.clicks) < 4 else "MANUAL: 4 corners"
        elif self.sel is not None:
            head = f"surface {self.sel + 1} selected{' (automatic choice)' if self.sel == self.auto else ''}"
        else:
            head = "no surface selected"
        lines = [head, "Enter save | 1-9 other surface | M click corners | R reset | Q quit"]
        if self.message:
            lines.append(self.message)
        for k, s in enumerate(lines):
            _text(img, s, (10, 22 + 24 * k), (255, 255, 255) if k != 2 else (0, 255, 255), 0.55)
        return img


def choose(bgr, points_cam, K, T_base_cam, max_range=2500.0):
    """Open the window; returns the chosen board, or None if the user quit."""
    ui = BinPicker(bgr, points_cam, K, T_base_cam, max_range)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, ui.on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW, ui.render())
            action = ui.handle_key(cv2.waitKey(30))
            if action == "quit" or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if action == "save":
                try:
                    return ui.board()
                except RuntimeError as exc:
                    ui.message = f"failed: {exc}"
    finally:
        cv2.destroyAllWindows()
