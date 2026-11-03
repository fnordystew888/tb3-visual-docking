"""Part B: measure the camera range/bearing error against ground truth to build the
EKF measurement noise model R(r), and check the camera extrinsics / frame chain.

The robot is placed at 600 random poses in the empty room. Every ArUco detection
is converted C -> B -> (range, bearing) by the same code the pipeline uses and
compared with the true range/bearing from the simulator. Errors are binned by
range; sigma_r(r) = a + b r^2 and a constant sigma_b are fitted.

Outputs: results/data/camera_noise_model.json, results/figures/camera_noise.png
Usage:   python scripts/characterise_sensors.py [--n 600] [--pixel-noise 0]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from tb3nav import scene as S
from tb3nav.sensors import Camera, true_range_bearing, base_pose, R_BC, T_BC

ROOT = os.path.join(os.path.dirname(__file__), "..")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--pixel-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    m = mujoco.MjModel.from_xml_string(S.build_xml(S.SceneConfig()))
    d = mujoco.MjData(m)
    cam = Camera(m, pixel_noise=a.pixel_noise, rng=rng)
    cam.delta_px = 0.0      # characterise the RAW pipeline, then fit the bias
    rows, raw = [], []
    for _ in range(a.n):
        x, y = rng.uniform(-2.2, 2.1), rng.uniform(-1.7, 1.7)
        if rng.random() < 0.3:           # bias some samples towards the dock (short range)
            x, y = rng.uniform(1.2, 2.2), rng.uniform(-0.4, 0.4)
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[:] = 0
        d.qpos[0:3] = [x, y, S.WHEEL_RADIUS + 0.0001]
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        pose = base_pose(d)
        for o in cam.observe(d):
            rt, bt = true_range_bearing(pose, o.marker_id)
            rows.append([o.marker_id, rt, bt, o.rng, o.bearing, o.pixels])
            raw.append(o.p_C_raw)
    A = np.array(rows)
    raw = np.array(raw)

    def corrected(delta):
        pc = raw * (A[:, 5] / (A[:, 5] + delta))[:, None]
        pb = pc @ R_BC.T + T_BC
        return np.hypot(pb[:, 0], pb[:, 1]), np.arctan2(pb[:, 1], pb[:, 0])

    er_raw = A[:, 3] - A[:, 1]
    deltas = np.linspace(-1, 1, 401)
    cost = [np.mean((corrected(dl)[0] - A[:, 1]) ** 2) for dl in deltas]
    delta = float(deltas[int(np.argmin(cost))])
    print(f"RAW range error mean {er_raw.mean()*1000:.2f} mm, std {er_raw.std()*1000:.2f} mm")
    print(f"fitted corner-shrink bias delta = {delta:.3f} px")
    A[:, 3], A[:, 4] = corrected(delta)
    er, eb = A[:, 3] - A[:, 1], wrap(A[:, 4] - A[:, 2])
    print(f"detections: {len(A)} from {a.n} poses")
    print(f"CORRECTED range error mean {er.mean()*1000:.2f} mm, std {er.std()*1000:.2f} mm, max |e| {np.abs(er).max()*1000:.1f} mm")
    print(f"bearing error mean {np.degrees(eb.mean()):.3f} deg, std {np.degrees(eb.std()):.3f} deg")

    # binned std vs range, then least-squares fit of sigma_r = a + b r^2
    bins = np.arange(0.0, 4.01, 0.25)
    idx = np.digitize(A[:, 1], bins)
    centres, sr, sb, cnt = [], [], [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if sel.sum() >= 8:
            centres.append(0.5 * (bins[k - 1] + bins[k]))
            sr.append(np.sqrt(np.mean(er[sel] ** 2)))   # RMS includes residual bias
            sb.append(np.sqrt(np.mean(eb[sel] ** 2)))
            cnt.append(int(sel.sum()))
    centres, sr, sb = map(np.array, (centres, sr, sb))
    M = np.vstack([np.ones_like(centres), centres ** 2]).T
    coef_r = np.linalg.lstsq(M, sr, rcond=None)[0]
    coef_r = np.maximum(coef_r, [0.002, 0.0])
    sigma_b = float(np.sqrt(np.mean(eb ** 2)))
    model = {"delta_px": delta, "raw_range_bias_mm": float(er_raw.mean() * 1000), "sigma_r_a": float(coef_r[0]), "sigma_r_b": float(coef_r[1]), "sigma_b": sigma_b,
             "inflation": 2.0,
             "note": "sigma_r(r) = a + b r^2 fitted to binned RMS error; EKF uses inflation x sigma",
             "n_detections": int(len(A)), "pixel_noise": a.pixel_noise,
             "bins": {"r": centres.tolist(), "rms_r": sr.tolist(), "rms_b": sb.tolist(), "count": cnt}}
    print("fitted:", {k: model[k] for k in ("sigma_r_a", "sigma_r_b", "sigma_b")})
    os.makedirs(os.path.join(ROOT, "results", "data"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "data", "camera_noise_model.json"), "w") as f:
        json.dump(model, f, indent=2)
    np.savetxt(os.path.join(ROOT, "results", "data", "camera_errors.csv"), A, delimiter=",",
               header="id,r_true,b_true,r_meas,b_meas,px", comments="")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].scatter(A[:, 1], er_raw * 1000, s=4, alpha=0.25, c="grey", label="raw")
    ax[0].scatter(A[:, 1], er * 1000, s=4, alpha=0.4, label="bias-corrected")
    rr = np.linspace(0.1, 4, 100)
    ax[0].plot(rr, 1000 * (coef_r[0] + coef_r[1] * rr ** 2), "r", label=r"fit $\sigma_r$")
    ax[0].plot(rr, -1000 * (coef_r[0] + coef_r[1] * rr ** 2), "r")
    ax[0].set(xlabel="true range [m]", ylabel="range error [mm]", title="Range error vs range")
    ax[0].legend()
    ax[1].scatter(A[:, 1], np.degrees(eb), s=4, alpha=0.4)
    ax[1].axhline(np.degrees(sigma_b), color="r"); ax[1].axhline(-np.degrees(sigma_b), color="r")
    ax[1].set(xlabel="true range [m]", ylabel="bearing error [deg]", title="Bearing error vs range")
    hist_all = np.histogram(A[:, 1], bins)[0]
    ax[2].bar(bins[:-1], hist_all, width=0.24, align="edge")
    ax[2].set(xlabel="true range [m]", ylabel="detections", title="Detections by range")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "camera_noise.png"), dpi=130)


if __name__ == "__main__":
    main()
