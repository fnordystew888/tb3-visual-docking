"""Part C: EKF fusing wheel-encoder odometry with camera range/bearing to known
ArUco landmarks (including the dock marker).

Prediction: odometry motion model driven by encoder increments. Process noise is
built in wheel space, M = diag(k |dphi_L| + s0^2, k |dphi_R| + s0^2), so that
uncertainty grows with distance travelled (slip / radius error scale with motion),
then mapped to the state with G J.

Update: range/bearing to a landmark with known world position. R depends on range
(measured in scripts/characterise_sensors.py). Innovations are gated with a
chi-square test (2 dof, 99 %: 9.21) to reject mis-detections and outliers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import model as Mdl
from . import scene as S
from .sensors import load_noise_model

CHI2_2DOF_99 = 9.21


@dataclass
class EKFParams:
    k_wheel: float = 2e-3          # wheel-angle variance per rad travelled [rad^2 / rad]
    s0_wheel: float = 1e-3         # floor on wheel-angle std per step [rad]
    r_scale: float = 10.0          # multiplies R; 10 chosen from the R sweep (NEES ~ 3), see README 'Tuning R'
    gate: float = CHI2_2DOF_99
    sigma_b_floor: float = 0.005   # [rad] floor on bearing std (unmodelled extrinsic / timing error)
    sigma_r_floor: float = 0.005   # [m]
    use_camera: bool = True
    # gate lock-out recovery: if markers are seen but EVERY innovation fails the gate
    # for this many consecutive camera frames, the estimate (not the camera) is
    # assumed wrong (e.g. wheel slip under a push) and P is inflated so the
    # measurements can pull the estimate back. 0 disables recovery.
    lockout_frames: int = 3
    reset_std: tuple = (0.5, 0.5, 0.6)   # [m, m, rad] added to P on recovery
    noise: dict = field(default_factory=load_noise_model)


class EKF:
    def __init__(self, x0, P0, params: EKFParams | None = None):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.p = params or EKFParams()
        self.J = Mdl.wheel_to_body_jacobian()
        self.n_upd = 0
        self.n_rej = 0
        self.last_nis = []
        self.consec_rej = 0
        self.n_resets = 0

    def R_of(self, r):
        n = self.p.noise
        infl = n.get("inflation", 2.0)
        sr = max(infl * (n["sigma_r_a"] + n["sigma_r_b"] * r * r), self.p.sigma_r_floor)
        sb = max(infl * n["sigma_b"], self.p.sigma_b_floor)
        return self.p.r_scale * np.diag([sr * sr, sb * sb])

    def predict(self, dphi_l, dphi_r):
        ds, dth = self.J @ np.array([dphi_l, dphi_r])
        F, G = Mdl.jacobians_odometry(self.x, ds, dth)
        Mw = np.diag([self.p.k_wheel * abs(dphi_l) + self.p.s0_wheel ** 2,
                      self.p.k_wheel * abs(dphi_r) + self.p.s0_wheel ** 2])
        Qx = G @ self.J @ Mw @ self.J.T @ G.T
        self.x = Mdl.f_discrete(self.x, ds, dth)
        self.P = F @ self.P @ F.T + Qx

    def update(self, obs_list, _retry=False):
        self.last_nis = []
        if not self.p.use_camera or not obs_list:
            return
        accepted = self._update(obs_list)
        if accepted:
            self.consec_rej = 0
        else:
            self.consec_rej += 1
            if self.p.lockout_frames and self.consec_rej >= self.p.lockout_frames and not _retry:
                self.P = self.P + np.diag(np.square(self.p.reset_std))
                self.n_resets += 1
                self.consec_rej = 0
                self.update(obs_list, _retry=True)

    def _update(self, obs_list):
        accepted = 0
        for o in obs_list:
            lm = S.LANDMARKS[o.marker_id][:2]
            z = np.array([o.rng, o.bearing])
            zhat = Mdl.h_range_bearing(self.x, lm)
            H = Mdl.H_range_bearing(self.x, lm)
            R = self.R_of(o.rng)
            y = z - zhat
            y[1] = Mdl.wrap(y[1])
            Sm = H @ self.P @ H.T + R
            nis = float(y @ np.linalg.solve(Sm, y))
            if nis > self.p.gate:
                self.n_rej += 1
                continue
            K = self.P @ H.T @ np.linalg.inv(Sm)
            self.x = self.x + K @ y
            self.x[2] = Mdl.wrap(self.x[2])
            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
            self.n_upd += 1
            accepted += 1
            self.last_nis.append(nis)
        return accepted
