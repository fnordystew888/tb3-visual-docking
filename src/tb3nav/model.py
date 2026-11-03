"""Part A: state-space model of the TurtleBot3 Burger (differential drive).

State     x = [p_x, p_y, theta]           (base pose in world frame W)
Input     u = [v, omega]                  (body linear / angular velocity)
Wheels    v = r (w_R + w_L) / 2,  omega = r (w_R - w_L) / b

Continuous-time kinematics (no-slip, rigid wheels, planar floor):
    p_x' = v cos(theta)
    p_y' = v sin(theta)
    theta' = omega

Tracking-error model used by the LQR (error expressed in the robot frame):
    e = R(theta)^T (p_ref - p),  e_theta = theta_ref - theta
    e1' =  omega e2 - v + v_r cos(e3)
    e2' = -omega e1 + v_r sin(e3)
    e3' =  omega_r - omega
With u1 = v_r cos(e3) - v and u2 = omega_r - omega, linearised about e = 0:
    A(v_r, w_r) = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],   B = [[1, 0], [0, 0], [0, 1]]
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from . import scene as S


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------- kinematics
def wheels_to_body(wl, wr, r=S.WHEEL_RADIUS, b=S.WHEEL_SEP):
    return r * (wr + wl) / 2, r * (wr - wl) / b


def body_to_wheels(v, w, r=S.WHEEL_RADIUS, b=S.WHEEL_SEP):
    return (v - w * b / 2) / r, (v + w * b / 2) / r


def f_discrete(x, ds, dth):
    """Odometry motion model (midpoint integration) for wheel increments."""
    th = x[2] + dth / 2
    return np.array([x[0] + ds * np.cos(th), x[1] + ds * np.sin(th), wrap(x[2] + dth)])


def jacobians_odometry(x, ds, dth):
    """F = df/dx, G = df/d[ds, dth]."""
    th = x[2] + dth / 2
    F = np.array([[1, 0, -ds * np.sin(th)], [0, 1, ds * np.cos(th)], [0, 0, 1]])
    G = np.array([[np.cos(th), -0.5 * ds * np.sin(th)], [np.sin(th), 0.5 * ds * np.cos(th)], [0, 1]])
    return F, G


def wheel_to_body_jacobian(r=S.WHEEL_RADIUS, b=S.WHEEL_SEP):
    """[ds, dth] = J [dphi_L, dphi_R]."""
    return np.array([[r / 2, r / 2], [-r / b, r / b]])


# ----------------------------------------------------------------- measurement
def h_range_bearing(x, lm):
    dx, dy = lm[0] - x[0], lm[1] - x[1]
    q = dx * dx + dy * dy
    return np.array([np.sqrt(q), wrap(np.arctan2(dy, dx) - x[2])])


def H_range_bearing(x, lm):
    dx, dy = lm[0] - x[0], lm[1] - x[1]
    q = dx * dx + dy * dy
    sq = np.sqrt(q)
    return np.array([[-dx / sq, -dy / sq, 0], [dy / q, -dx / q, -1]])


# ----------------------------------------------------------------- linear analysis
def error_model(v_r, w_r):
    A = np.array([[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0.0]])
    B = np.array([[1, 0], [0, 0], [0, 1.0]])
    return A, B


def discretise(A, B, dt):
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])


def obsv(A, C):
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def lqr_gain(v_r, w_r, dt, Q, R):
    """Discrete LQR gain for the error model linearised at (v_r, w_r).
    Returns K such that u = -K e."""
    A, B = discretise(*error_model(v_r, w_r), dt)
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
