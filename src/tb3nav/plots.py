"""Plotting helpers for trial logs."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Ellipse

from . import scene as S
from . import model as Mdl


def draw_room(ax, obstacles, grid=None):
    ax.add_patch(Rectangle((-S.ROOM_X, -S.ROOM_Y), 2 * S.ROOM_X, 2 * S.ROOM_Y, fill=False, lw=2))
    if grid is not None:
        from .planner import RES
        ax.imshow(grid.T, origin="lower", extent=(-S.ROOM_X, S.ROOM_X, -S.ROOM_Y, S.ROOM_Y),
                  cmap="Greys", alpha=0.35, vmin=0, vmax=1.5)
    for kind, x, y, s in obstacles:
        if kind == "box":
            ax.add_patch(Rectangle((x - s, y - s), 2 * s, 2 * s, color="tab:brown", alpha=0.6))
        else:
            ax.add_patch(Circle((x, y), s, color="tab:blue", alpha=0.5))
    ax.add_patch(Rectangle((S.DOCK_FACE_X, -0.15), S.ROOM_X - S.DOCK_FACE_X, 0.3, color="tab:green"))
    for mid, (mx, my, _, _) in S.LANDMARKS.items():
        ax.plot(mx, my, "ks" if mid else "g*", ms=6)
        ax.annotate(str(mid), (mx, my), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlim(-S.ROOM_X - 0.1, S.ROOM_X + 0.1)
    ax.set_ylim(-S.ROOM_Y - 0.1, S.ROOM_Y + 0.1)


def cov_ellipse(ax, mu, P, nsig=3, **kw):
    w, V = np.linalg.eigh(P[:2, :2])
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mu[:2], 2 * nsig * np.sqrt(max(w[1], 0)), 2 * nsig * np.sqrt(max(w[0], 0)), angle=ang, fill=False, **kw))


def trial_overview(log, path_out, title=""):
    T, E, P, t = log["true"], log["est"], log["P"], log["t"]
    err = E - T
    err[:, 2] = Mdl.wrap(err[:, 2])
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[:, 0])
    draw_room(ax, log["obstacles"], log.get("grid"))
    ax.plot(log["path"][:, 0], log["path"][:, 1], "c--", lw=1, label="final planned path")
    ax.plot(T[:, 0], T[:, 1], "k", lw=1.6, label="true")
    ax.plot(E[:, 0], E[:, 1], "r", lw=1, label="EKF estimate")
    for k in range(0, len(E), 40):
        cov_ellipse(ax, E[k], P[k], ec="r", lw=0.6)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or "Trajectory (3-sigma ellipses every 2 s)")
    labels = ["x error [m]", "y error [m]", "yaw error [deg]"]
    for i in range(3):
        a = fig.add_subplot(gs[i, 1])
        sc = np.degrees(1) if i == 2 else 1
        s3 = 3 * np.sqrt(P[:, i, i]) * sc
        a.fill_between(t, -s3, s3, color="r", alpha=0.15, label=r"$\pm3\sigma$")
        a.plot(t, err[:, i] * sc, "k", lw=1, label="estimate - truth")
        seen = log["n_obs"] > 0
        a.scatter(t[seen], np.full(seen.sum(), -s3.max() * 0.95), s=2, c="g", label="marker seen" if i == 0 else None)
        a.set_ylabel(labels[i])
        if i == 0:
            a.legend(fontsize=7, loc="upper right", ncol=3)
    a.set_xlabel("time [s]")
    fig.suptitle(f"{log['metrics']['status']}: dock error {log['metrics']['dock_pos_err_m']*100:.1f} cm, "
                 f"est RMSE {log['metrics']['est_rmse_pos_m']*100:.1f} cm, NEES {log['metrics']['nees_mean']:.2f}")
    fig.tight_layout()
    fig.savefig(path_out, dpi=120)
    plt.close(fig)
