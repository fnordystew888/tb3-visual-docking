"""Fast checks of the maths that everything else relies on. Run: python -m pytest tests -q"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tb3nav import model as M
from tb3nav import scene as S
from tb3nav.sensors import R_BC


def num_jac(f, x, eps=1e-6):
    fx = f(x)
    J = np.zeros((len(fx), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x)); dx[i] = eps
        d = f(x + dx) - fx
        d = np.array([M.wrap(v) if k == len(fx) - 1 and len(fx) == 3 else v for k, v in enumerate(d)])
        J[:, i] = d / eps
    return J


def test_odometry_jacobian():
    x = np.array([0.3, -0.2, 0.7])
    F, G = M.jacobians_odometry(x, 0.05, 0.1)
    assert np.allclose(F, num_jac(lambda z: M.f_discrete(z, 0.05, 0.1), x), atol=1e-5)
    assert np.allclose(G, num_jac(lambda u: M.f_discrete(x, u[0], u[1]), np.array([0.05, 0.1])), atol=1e-5)


def test_measurement_jacobian():
    x, lm = np.array([0.1, 0.2, 0.4]), np.array([1.5, 1.0])
    Hn = num_jac(lambda z: M.h_range_bearing(z, lm), x)
    assert np.allclose(M.H_range_bearing(x, lm), Hn, atol=1e-5)


def test_controllability():
    assert np.linalg.matrix_rank(M.ctrb(*M.error_model(0.15, 0.0))) == 3
    assert np.linalg.matrix_rank(M.ctrb(*M.error_model(0.0, 0.0))) == 2


def test_lqr_stable():
    K = M.lqr_gain(0.15, 0.3, 0.05, np.diag([4, 25, 2]), np.diag([1, 0.08]))
    Ad, Bd = M.discretise(*M.error_model(0.15, 0.3), 0.05)
    assert np.all(np.abs(np.linalg.eigvals(Ad - Bd @ K)) < 1)


def test_camera_frame():
    # optical axis (z_C) must be the robot's forward axis, x_C must point to the robot's right
    assert np.allclose(R_BC @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(R_BC @ [1, 0, 0], [0, -1, 0])
    assert np.isclose(np.linalg.det(R_BC), 1)


def test_wheel_mapping_roundtrip():
    wl, wr = M.body_to_wheels(0.12, 0.8)
    v, w = M.wheels_to_body(wl, wr)
    assert np.isclose(v, 0.12) and np.isclose(w, 0.8)
    assert S.WHEEL_RADIUS == 0.033 and S.WHEEL_SEP == 0.160
