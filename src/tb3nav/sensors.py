"""Sensor models: wheel encoders, 2D LiDAR and RGB camera with ArUco detection.

Frames (see docs/frames.md):
    world  W : room frame, origin at room centre, z up.
    base   B : robot base_link, x forward, y left, z up, origin midway between wheels on the axle.
    laser  L : same orientation as B, origin (0, 0, 0.17) in B.
    camera C : OpenCV optical frame, x right, y down, z forward, origin CAM_POS_BASE in B.
    marker M : centre of the ArUco square.

The camera returns marker centres in C via solvePnP; they are transformed C -> B
here and reduced to (range, bearing) in the B frame, which is what the EKF uses.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from . import scene as S

TICKS_PER_REV = 4096  # Dynamixel XL430 encoder resolution

# Rotation from camera optical frame (C) to base frame (B):
#   x_B = z_C, y_B = -x_C, z_B = -y_C
R_BC = np.array([[0.0, 0.0, 1.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0]])
T_BC = S.CAM_POS_BASE.copy()

CALIB_PATH = os.path.join(os.path.dirname(S.ASSET_DIR), "results", "data", "camera_calib.json")
NOISE_PATH = os.path.join(os.path.dirname(S.ASSET_DIR), "results", "data", "camera_noise_model.json")


def load_noise_model():
    """Measured camera error model from scripts/characterise_sensors.py."""
    if os.path.exists(NOISE_PATH):
        with open(NOISE_PATH) as f:
            return json.load(f)
    return {"delta_px": 0.0, "sigma_r_a": 0.01, "sigma_r_b": 0.01, "sigma_b": 0.005, "inflation": 2.0}


def nominal_intrinsics(w=S.CAM_W, h=S.CAM_H, fovy=S.CAM_FOVY):
    """Pinhole intrinsics implied by MuJoCo's vertical field of view (square pixels)."""
    f = 0.5 * h / np.tan(np.radians(fovy) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def load_calibration():
    """Calibrated K and distortion from scripts/calibrate_camera.py; falls back to the
    nominal model only if calibration has not been run."""
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            c = json.load(f)
        return np.array(c["K"]), np.array(c["dist"]), True
    return nominal_intrinsics(), np.zeros(5), False


class Encoders:
    """Quantised wheel encoders. Returns wheel angles [rad] (left, right)."""

    def __init__(self, model, extra_noise_std=0.0, rng=None):
        self.adr = [model.sensor("enc_l").adr[0], model.sensor("enc_r").adr[0]]
        self.q = 2 * np.pi / TICKS_PER_REV
        self.extra = extra_noise_std
        self.rng = rng or np.random.default_rng()
        self.offset = np.zeros(2)

    def read(self, data):
        raw = np.array([data.sensordata[self.adr[0]], data.sensordata[self.adr[1]]])
        if self.extra > 0:
            # random-walk noise per read models slip / backlash not captured by the contact model
            self.offset += self.rng.normal(0, self.extra, 2)
        return np.round((raw + self.offset) / self.q) * self.q


@dataclass
class LidarScan:
    angles: np.ndarray   # beam angles in laser frame [rad]
    ranges: np.ndarray   # [m], np.inf for no return


class Lidar:
    """LDS-01-like scanner. Noise model from the LDS-01 datasheet
    (accuracy about +-15 mm below 0.5 m, +-5 % beyond, here sigma = 0.01 + 0.01 r)."""

    def __init__(self, model, sigma0=0.01, sigma_rel=0.01, dropout=0.01, rng=None):
        self.adr = np.array([model.sensor(f"r{k}").adr[0] for k in range(S.LIDAR_BEAMS)])
        self.angles = 2 * np.pi * np.arange(S.LIDAR_BEAMS) / S.LIDAR_BEAMS
        self.s0, self.sr, self.drop = sigma0, sigma_rel, dropout
        self.rng = rng or np.random.default_rng()

    def read(self, data) -> LidarScan:
        r = data.sensordata[self.adr].copy()
        valid = (r > 0.12) & (r < S.LIDAR_RANGE)
        noisy = r + self.rng.normal(0, 1, r.shape) * (self.s0 + self.sr * r)
        noisy[~valid] = np.inf
        noisy[self.rng.random(r.shape) < self.drop] = np.inf
        return LidarScan(self.angles.copy(), noisy)


@dataclass
class MarkerObs:
    marker_id: int
    rng: float            # planar range in B [m]
    bearing: float        # bearing in B [rad]
    p_B: np.ndarray       # 3D marker centre in B (bias-corrected)
    pixels: float         # apparent side length [px] (used for gating / R model)
    p_C_raw: np.ndarray = None  # uncorrected solvePnP translation in C


class Camera:
    """Renders the onboard camera, detects ArUco markers and returns their
    range/bearing in the base frame."""

    def __init__(self, model, pixel_noise=0.0, blur=0, rng=None, K=None, dist=None):
        self.renderer = mujoco.Renderer(model, S.CAM_H, S.CAM_W)
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        if K is None:
            K, dist, _ = load_calibration()
        self.K, self.dist = K, dist
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(d, p)
        self.pixel_noise, self.blur = pixel_noise, blur
        self.rng = rng or np.random.default_rng()
        self.last_image = None
        # corner-shrink bias [px]: ArUco corners on a rendered (anti-aliased) edge sit
        # about 0.3 px inside the true square, so small markers read as too far away.
        self.delta_px = load_noise_model().get("delta_px", 0.0)

    def render(self, data):
        self.renderer.update_scene(data, camera="rgb", scene_option=self.opt)
        img = self.renderer.render()
        if self.blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.blur)
        if self.pixel_noise > 0:
            img = np.clip(img.astype(np.float32) + self.rng.normal(0, self.pixel_noise, img.shape), 0, 255).astype(np.uint8)
        self.last_image = img
        return img

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        obs = []
        if ids is None:
            return obs, corners, ids
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in S.LANDMARKS:
                continue
            side = S.marker_black_side(mid)
            h = side / 2
            objp = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(objp, c.reshape(4, 2).astype(np.float64), self.K, self.dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            p_C_raw = tvec.flatten()
            px = float(np.mean(np.linalg.norm(np.diff(np.vstack([c.reshape(4, 2), c.reshape(4, 2)[:1]]), axis=0), axis=1)))
            p_C = p_C_raw * px / (px + self.delta_px)          # range-bias calibration
            p_B = R_BC @ p_C + T_BC                              # C -> B
            obs.append(MarkerObs(mid, float(np.hypot(p_B[0], p_B[1])), float(np.arctan2(p_B[1], p_B[0])), p_B, px, p_C_raw))
        return obs, corners, ids

    def observe(self, data):
        return self.detect(self.render(data))[0]


def true_range_bearing(pose, marker_id):
    """Ground-truth planar range/bearing of a marker centre from the robot base
    (evaluation and characterisation only)."""
    mx, my = S.LANDMARKS[marker_id][:2]
    dx, dy = mx - pose[0], my - pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    bx, by = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(bx, by), np.arctan2(by, bx)


def base_pose(data):
    """Ground-truth (x, y, yaw) of the base. Used ONLY for evaluation/logging."""
    q = data.qpos[3:7]
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([data.qpos[0], data.qpos[1], yaw])
