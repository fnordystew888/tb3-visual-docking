"""Record demo clips for the video: top-down view with the true pose (green), the
EKF estimate (red) with its 3-sigma ellipse, the planned path and the LiDAR
occupancy grid, next to the onboard camera with detected markers.

Usage: python scripts/run_demo.py [--condition nominal|disturbed|blind] [--seed 7]
Output: results/video/<condition>.mp4 and results/figures/demo_<condition>.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tb3nav  # noqa: F401  (sets the GL backend on Linux; must precede `import mujoco`)
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from tb3nav import scene as S
from tb3nav.planner import RES
from tb3nav.plots import trial_overview
from tb3nav.sim import TrialConfig, run_trial

ROOT = os.path.join(os.path.dirname(__file__), "..")
TOP_H, TOP_W = 480, 640
F_TOP = (TOP_H / 2) / np.tan(np.radians(45 / 2))
Z_TOP = 6.2


def w2px(x, y, z=0.0):
    return int(TOP_W / 2 + F_TOP * x / (Z_TOP - z)), int(TOP_H / 2 - F_TOP * y / (Z_TOP - z))


class Recorder:
    def __init__(self, title):
        self.frames, self.r, self.title = [], None, title
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.trail_true, self.trail_est = [], []

    def __call__(self, d, img, obs, ekf, ctrl, grid, gt):
        m = d.model if hasattr(d, "model") else None
        if self.r is None:
            self.r = self.shared_renderer   # one GL context per process (OSMesa)
        self.r.update_scene(d, camera="top", scene_option=self.opt)
        top = self.r.render().copy()
        # LiDAR-discovered occupied cells (not the known walls)
        occ = (grid.logodds > 1.5) & ~grid.known
        for i, j in zip(*np.nonzero(occ)):
            x, y = grid.cell_centre(i, j)
            cv2.circle(top, w2px(x, y), 2, (255, 140, 0), -1)
        P = ctrl.path
        pts = np.array([w2px(x, y) for x, y in P[:, :2]], np.int32)
        cv2.polylines(top, [pts], False, (0, 200, 255), 1)
        self.trail_true.append(w2px(gt[0], gt[1]))
        self.trail_est.append(w2px(ekf.x[0], ekf.x[1]))
        cv2.polylines(top, [np.array(self.trail_true, np.int32)], False, (0, 160, 0), 2)
        cv2.polylines(top, [np.array(self.trail_est, np.int32)], False, (220, 0, 0), 1)
        # 3-sigma ellipse
        wv, V = np.linalg.eigh(ekf.P[:2, :2])
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ax = (max(int(3 * np.sqrt(max(wv[1], 0)) * F_TOP / Z_TOP), 1), max(int(3 * np.sqrt(max(wv[0], 0)) * F_TOP / Z_TOP), 1))
        cv2.ellipse(top, w2px(ekf.x[0], ekf.x[1]), ax, -ang, 0, 360, (220, 0, 0), 1)
        hx = ekf.x[:2] + 0.2 * np.array([np.cos(ekf.x[2]), np.sin(ekf.x[2])])
        cv2.arrowedLine(top, w2px(*ekf.x[:2]), w2px(*hx), (220, 0, 0), 2, tipLength=0.3)
        # marker rays
        for o in obs:
            mx, my = S.LANDMARKS[o.marker_id][:2]
            cv2.line(top, w2px(ekf.x[0], ekf.x[1]), w2px(mx, my), (255, 0, 255), 1)
        cam = img.copy()
        _, corners, ids = self.cam_detect(img)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cam, corners, ids)
        err = np.hypot(*(ekf.x[:2] - gt[:2]))
        s = np.sqrt(np.trace(ekf.P[:2, :2]))
        txt = [f"{self.title}   t = {d.time:5.1f} s",
               f"est ({ekf.x[0]:+.2f}, {ekf.x[1]:+.2f}, {np.degrees(ekf.x[2]):+.0f} deg)",
               f"true ({gt[0]:+.2f}, {gt[1]:+.2f}, {np.degrees(gt[2]):+.0f} deg)",
               f"|est - true| = {err*100:4.1f} cm   sigma_pos = {s*100:4.1f} cm",
               f"markers in view: {len(obs)}"]
        panel = np.hstack([top, cam])
        bar = np.full((110, panel.shape[1], 3), 255, np.uint8)
        for k, t in enumerate(txt):
            cv2.putText(bar, t, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bar, "green: true   red: EKF + 3 sigma   cyan: A* path   orange: LiDAR obstacles", (660, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        self.frames.append(np.vstack([panel, bar]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="nominal", choices=["nominal", "disturbed", "blind"])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.condition == "nominal":
        cfg, title = TrialConfig(seed=a.seed), "NOMINAL"
    elif a.condition == "disturbed":
        cfg = TrialConfig(seed=a.seed, push_force=8.0, push_time=12.0, wheel_radius_err=0.03, floor_friction=0.6,
                          occlusion_prob=0.5, pixel_noise=10.0, blur=1.0)
        title = "DISTURBED: 8 N push @12 s, +3% wheel radius, 50% frame drop, image noise"
    else:
        cfg, title = TrialConfig(seed=a.seed, occlusion_window=(8.0, 16.0)), "CAMERA BLIND 8-16 s"
    cfg.video_every = 1
    rec = Recorder(title)

    # the recorder needs the model and a detector; wrap run_trial's callback
    from tb3nav import sim as SIM
    orig = SIM.Camera

    class CamHook(orig):
        def __init__(self, model, **kw):
            super().__init__(model, **kw)
            rec.model = model
            rec.cam_detect = self.detect
            rec.shared_renderer = self.renderer
    SIM.Camera = CamHook
    log = run_trial(cfg, frame_cb=rec)
    SIM.Camera = orig
    os.makedirs(os.path.join(ROOT, "results", "video"), exist_ok=True)
    out = os.path.join(ROOT, "results", "video", f"{a.condition}.mp4")
    imageio.mimsave(out, rec.frames, fps=10, macro_block_size=1)
    trial_overview(log, os.path.join(ROOT, "results", "figures", f"demo_{a.condition}.png"), title=title)
    print(out, len(rec.frames), "frames", log["metrics"])


if __name__ == "__main__":
    main()
