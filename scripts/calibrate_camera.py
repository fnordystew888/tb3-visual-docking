"""Part B: intrinsic calibration of the simulated onboard camera.

A 9x7-square checkerboard (30 mm squares) is shown to a camera with the same
resolution and field of view as the robot camera at 40 random poses, and
cv2.calibrateCamera is run on the detected corners. The result is compared with
the pinhole model implied by MuJoCo's fovy, and saved to
results/data/camera_calib.json, which the pipeline loads at runtime.

Usage: python scripts/calibrate_camera.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tb3nav import scene as S
from tb3nav.sensors import CALIB_PATH, nominal_intrinsics

SQ = 0.03            # square size [m]
NX, NY = 9, 7        # squares
PX = 40              # texture pixels per square


def board_png():
    img = np.full((NY * PX + 2 * PX, NX * PX + 2 * PX), 255, np.uint8)
    for j in range(NY):
        for i in range(NX):
            if (i + j) % 2 == 0:
                img[PX + j * PX:PX + (j + 1) * PX, PX + i * PX:PX + (i + 1) * PX] = 0
    p = os.path.join(S.ASSET_DIR, "checkerboard.png")
    os.makedirs(S.ASSET_DIR, exist_ok=True)
    cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return p


def main(n_views=40, seed=0):
    rng = np.random.default_rng(seed)
    hw, hh = (NX + 2) * SQ / 2, (NY + 2) * SQ / 2
    xml = f"""<mujoco><visual><global offwidth="{S.CAM_W}" offheight="{S.CAM_H}"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/></visual>
    <asset><texture name="cb" type="2d" file="{board_png()}"/>
    <material name="cb" texture="cb" texrepeat="1 1" texuniform="false"/></asset>
    <worldbody>
      <camera name="rgb" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{S.CAM_FOVY}"/>
      <body name="board" mocap="true"><geom type="plane" size="{hw} {hh} 0.01" material="cb" euler="0 90 0" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, S.CAM_H, S.CAM_W)

    objp = np.zeros(((NX - 1) * (NY - 1), 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX - 1, 0:NY - 1].T.reshape(-1, 2) * SQ
    obj_pts, img_pts = [], []
    tries = 0
    while len(obj_pts) < n_views and tries < 400:
        tries += 1
        dist = rng.uniform(0.35, 0.9)
        pos = np.array([dist, rng.uniform(-0.25, 0.25) * dist, rng.uniform(-0.18, 0.18) * dist])
        # board faces the camera (normal -x) with random tilt up to ~35 deg
        rot = Rotation.from_euler("zyx", [np.pi + rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)])
        q = rot.as_quat()  # x y z w
        d.mocap_pos[0] = pos
        d.mocap_quat[0] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="rgb")
        gray = cv2.cvtColor(r.render(), cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (NX - 1, NY - 1))
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        obj_pts.append(objp)
        img_pts.append(corners)
        if len(obj_pts) == 1:
            vis = cv2.drawChessboardCorners(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (NX - 1, NY - 1), corners, ok)
            os.makedirs(os.path.join(os.path.dirname(CALIB_PATH), "..", "figures"), exist_ok=True)
            cv2.imwrite(os.path.join(os.path.dirname(CALIB_PATH), "..", "figures", "calib_view.png"), vis)

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (S.CAM_W, S.CAM_H), None, None)
    Kn = nominal_intrinsics()
    print(f"views used: {len(obj_pts)} / {tries} rendered")
    print(f"reprojection RMS: {rms:.3f} px")
    print("calibrated K:\n", np.round(K, 2))
    print("nominal K (from fovy):\n", np.round(Kn, 2))
    print("distortion:", np.round(dist.flatten(), 4))
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump({"K": K.tolist(), "dist": dist.flatten().tolist(), "rms_px": rms, "n_views": len(obj_pts),
                   "K_nominal": Kn.tolist(), "square_m": SQ, "board_inner_corners": [NX - 1, NY - 1]}, f, indent=2)
    print("saved", os.path.abspath(CALIB_PATH))


if __name__ == "__main__":
    main()
