"""Part D: gain-scheduled discrete LQR path tracker running on the EKF estimate.

At each control step the reference is the closest point on the path ahead of the
robot (searched in a window so the index only moves forward). The tracking error
e = [e1, e2, e3] in the robot frame is fed to u = -K(v_r, w_r) e where K is the
discrete LQR gain of the linearised error model at the current reference speed
and curvature. Gains are cached on a (v_r, w_r) grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model as Mdl
from . import scene as S


@dataclass
class LQRParams:
    q_long: float = 4.0      # weight on along-track error e1 [1/m^2]
    q_lat: float = 25.0      # weight on cross-track error e2 [1/m^2]
    q_head: float = 2.0      # weight on heading error e3 [1/rad^2]
    r_v: float = 1.0         # weight on u1 [s^2/m^2]
    r_w: float = 0.08        # weight on u2 [s^2/rad^2]
    v_ref: float = 0.15      # cruise speed [m/s] (68 % of TB3 max, leaves authority for corrections)
    dt: float = 0.05         # control period [s]
    lookahead: float = 0.0   # reference offset ahead of the closest-point projection [m]
    v_min: float = 0.03      # progress floor [m/s] while heading error < 60 deg (see 'LQR deadlock' in README)


class LQRTracker:
    def __init__(self, params: LQRParams | None = None):
        self.p = params or LQRParams()
        self.Q = np.diag([self.p.q_long, self.p.q_lat, self.p.q_head])
        self.R = np.diag([self.p.r_v, self.p.r_w])
        self._cache = {}
        self.path = None
        self.idx = 0
        self.last_e = np.zeros(3)
        self.last_K = None

    def set_path(self, path):
        self.path = path
        self.idx = 0

    def gain(self, v_r, w_r):
        key = (round(max(v_r, 0.02), 2), round(w_r, 1))
        if key not in self._cache:
            self._cache[key] = Mdl.lqr_gain(key[0], key[1], self.p.dt, self.Q, self.R)
        return self._cache[key]

    def remaining(self):
        return self.path[-1, 4] - self.path[self.idx, 4]

    def step(self, x_hat):
        P = self.path
        win = slice(self.idx, min(self.idx + 40, len(P)))
        d = np.hypot(P[win, 0] - x_hat[0], P[win, 1] - x_hat[1])
        self.idx = self.idx + int(np.argmin(d))
        # reference slightly ahead
        s_ref = min(P[self.idx, 4] + self.p.lookahead, P[-1, 4])
        k = min(int(np.searchsorted(P[:, 4], s_ref)), len(P) - 1)
        xr, yr, thr, kap = P[k, 0], P[k, 1], P[k, 2], P[k, 3]
        rem = P[-1, 4] - P[self.idx, 4]
        v_r = float(np.clip(min(self.p.v_ref, 0.6 * rem + 0.02), 0.0, self.p.v_ref))
        w_r = float(np.clip(v_r * kap, -S.W_MAX, S.W_MAX))
        c, s = np.cos(x_hat[2]), np.sin(x_hat[2])
        dx, dy = xr - x_hat[0], yr - x_hat[1]
        e = np.array([c * dx + s * dy, -s * dx + c * dy, Mdl.wrap(thr - x_hat[2])])
        K = self.gain(v_r, w_r)
        u = -K @ e
        v = v_r * np.cos(e[2]) - u[0]
        w = w_r - u[1]
        # Progress floor. K is 2x3, so the linear law has a 1-D null space: an error
        # e != 0 with v_r cos(e3) - K1 e = 0 and w_r - K2 e = 0 stops the robot
        # while the closest-point reference stays put -> permanent stall. Observed
        # in 2 of 241 trials; the floor guarantees the reference advances.
        if abs(e[2]) < np.radians(60) and rem > 0.01:
            v = max(v, min(self.p.v_min, v_r))
        v = float(np.clip(v, -0.05, S.V_MAX))
        w = float(np.clip(w, -S.W_MAX, S.W_MAX))
        self.last_e, self.last_K = e, K
        return v, w, e
