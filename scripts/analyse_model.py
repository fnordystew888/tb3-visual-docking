"""Part A: controllability / observability analysis and LQR design checks.

1. Controllability of the linearised tracking-error model (A(v_r, w_r), B) over the
   operating envelope, including the degenerate v_r = w_r = 0 case.
2. Local observability of the EKF's linearised measurement model for the
   sensing situations the robot actually meets: no marker, one marker while
   stationary, one marker while moving, two markers.
3. LQR gains and discrete closed-loop poles at cruise and docking speed.

Outputs: results/data/model_analysis.json, results/figures/controllability.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tb3nav import model as M
from tb3nav import scene as S
from tb3nav.controller import LQRParams

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    out = {}
    # ---------------------------------------------------------------- controllability
    cases = {"cruise (v=0.15, w=0)": (0.15, 0.0), "turning (v=0.15, w=0.6)": (0.15, 0.6),
             "turn on spot (v=0, w=0.5)": (0.0, 0.5), "stopped (v=0, w=0)": (0.0, 0.0)}
    out["controllability"] = {}
    for name, (v, w) in cases.items():
        A, B = M.error_model(v, w)
        C = M.ctrb(A, B)
        sv = np.linalg.svd(C, compute_uv=False)
        out["controllability"][name] = {"rank": int(np.linalg.matrix_rank(C, tol=1e-9)), "min_singular_value": float(sv[-1])}
        print(f"ctrb {name:28s} rank {np.linalg.matrix_rank(C, tol=1e-9)}  sigma_min {sv[-1]:.4f}")

    vs = np.linspace(0, 0.22, 60)
    smin = [np.linalg.svd(M.ctrb(*M.error_model(v, 0.0)), compute_uv=False)[-1] for v in vs]

    # ---------------------------------------------------------------- observability
    x = np.array([0.0, 0.0, 0.3])
    lm1, lm2 = np.array(S.LANDMARKS[5][:2]), np.array(S.LANDMARKS[7][:2])
    dt, v = 0.05, 0.15

    def obs_rank(H_list, F_list):
        rows, Phi = [], np.eye(3)
        for H, F in zip(H_list, F_list):
            rows.append(H @ Phi)
            Phi = F @ Phi
        O = np.vstack(rows)
        return int(np.linalg.matrix_rank(O, tol=1e-9)), float(np.linalg.svd(O, compute_uv=False)[-1])

    F_still = [np.eye(3)] * 10
    ds = v * dt
    F_move, xs = [], [x.copy()]
    for k in range(10):
        F, _ = M.jacobians_odometry(xs[-1], ds, 0.0)
        F_move.append(F)
        xs.append(M.f_discrete(xs[-1], ds, 0.0))
    scen = {
        "no marker": (np.zeros((0, 3)), None),
        "1 marker, stationary": ([M.H_range_bearing(x, lm1)] * 10, F_still),
        "1 marker, moving": ([M.H_range_bearing(xk, lm1) for xk in xs[:10]], F_move),
        "marker A then marker B, moving": ([M.H_range_bearing(xk, lm1 if k < 5 else lm2) for k, xk in enumerate(xs[:10])], F_move),
        "2 markers, stationary": ([np.vstack([M.H_range_bearing(x, lm1), M.H_range_bearing(x, lm2)])] * 10, F_still),
    }
    out["observability"] = {}
    for name, (Hs, Fs) in scen.items():
        if Fs is None:
            r, s = 0, 0.0
        else:
            r, s = obs_rank(Hs, Fs)
        out["observability"][name] = {"rank": r, "min_singular_value": s}
        print(f"obsv {name:24s} rank {r}  sigma_min {s:.5f}")
    # Single-marker stationary null space: which direction is unobservable?
    H = M.H_range_bearing(x, lm1)
    null = np.linalg.svd(H)[2][-1]
    out["observability"]["1 marker, stationary"]["null_space"] = null.tolist()
    print("  unobservable direction (dx, dy, dtheta):", np.round(null, 3),
          "= rotating about the marker while keeping range and bearing constant")

    # ---------------------------------------------------------------- LQR
    p = LQRParams()
    Q, R = np.diag([p.q_long, p.q_lat, p.q_head]), np.diag([p.r_v, p.r_w])
    out["lqr"] = {"Q": np.diag(Q).tolist(), "R": np.diag(R).tolist(), "dt": p.dt, "cases": {}}
    for name, (v, w) in {"cruise": (p.v_ref, 0.0), "dock": (0.08, 0.0), "turn": (p.v_ref, 0.6)}.items():
        K = M.lqr_gain(v, w, p.dt, Q, R)
        Ad, Bd = M.discretise(*M.error_model(v, w), p.dt)
        poles = np.linalg.eigvals(Ad - Bd @ K)
        tc = -p.dt / np.log(np.abs(poles))
        out["lqr"]["cases"][name] = {"K": K.round(4).tolist(), "poles": [complex(z) for z in poles], "time_constants_s": tc.tolist()}
        print(f"LQR {name}: K =\n{K.round(3)}\n  |poles| {np.abs(poles).round(4)}  time constants {tc.round(2)} s")
    for c in out["lqr"]["cases"].values():
        c["poles"] = [[z.real, z.imag] for z in c["poles"]]

    with open(os.path.join(ROOT, "results", "data", "model_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(vs, smin)
    ax.set(xlabel=r"reference speed $v_r$ [m/s] ($\omega_r = 0$)", ylabel=r"$\sigma_{min}(\mathcal{C})$",
           title="Controllability of the tracking-error model")
    ax.axvline(0.08, ls=":", c="g", label="dock speed")
    ax.axvline(0.15, ls="--", c="k", label="cruise speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "controllability.png"), dpi=130)


if __name__ == "__main__":
    main()
