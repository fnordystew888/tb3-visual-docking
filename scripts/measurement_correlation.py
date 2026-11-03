"""Part B/C diagnosis: are consecutive camera measurement errors independent?

The EKF assumes white measurement noise. The static characterisation
(characterise_sensors.py) measures the error *size* but not its correlation in
time. Here we drive nominal trials and record the range/bearing error of every
detection against ground truth, then compute the autocorrelation of the error
for each continuous track of a marker (same id seen in consecutive frames).

If errors are strongly correlated over N frames, the filter effectively counts
the same error N times and becomes overconfident by about N, which is what a
NEES well above 3 indicates. The effective number of independent samples is
N_eff = N (1 - rho) / (1 + rho) for an AR(1) process with lag-1 correlation rho.

Output: results/data/measurement_correlation.json, results/figures/meas_autocorr.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tb3nav  # noqa: F401
from tb3nav.model import wrap
from tb3nav.sensors import true_range_bearing
from tb3nav.sim import TrialConfig, run_trial

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main(seeds=(2000, 2001, 2002, 2003)):
    tracks = {}
    for sd in seeds:
        frame = [0]

        def cb(d, img, obs, ekf, ctrl, grid, gt):
            for o in obs:
                rt, bt = true_range_bearing(gt, o.marker_id)  # GROUND TRUTH (evaluation only)
                tracks.setdefault((sd, o.marker_id), []).append((frame[0], o.rng - rt, wrap(o.bearing - bt), rt))
            frame[0] += 1

        cfg = TrialConfig(seed=sd)
        cfg.video_every = 1
        run_trial(cfg, frame_cb=cb)

    maxlag = 15
    acf_r, acf_b, weights = np.zeros(maxlag + 1), np.zeros(maxlag + 1), np.zeros(maxlag + 1)
    for key, rows in tracks.items():
        A = np.array(rows)
        # split into runs of consecutive frames
        breaks = np.where(np.diff(A[:, 0]) != 1)[0] + 1
        for run in np.split(A, breaks):
            if len(run) < maxlag + 5:
                continue
            for col, acc in ((1, acf_r), (2, acf_b)):
                # about ZERO, not the run mean: a bias that persists for the whole
                # track is exactly the correlated error the EKF cannot average out
                x = run[:, col]
                v = np.mean(x * x)
                if v <= 0:
                    continue
                for L in range(maxlag + 1):
                    acc[L] += len(run) * np.mean(x[L:] * x[:len(x) - L]) / v
            weights += len(run)
    acf_r /= np.maximum(weights, 1)
    acf_b /= np.maximum(weights, 1)
    rho_r, rho_b = acf_r[1], acf_b[1]
    out = {"lag1_range": float(rho_r), "lag1_bearing": float(rho_b),
           "neff_ratio_range": float((1 - rho_r) / (1 + rho_r)), "neff_ratio_bearing": float((1 - rho_b) / (1 + rho_b)),
           "acf_range": acf_r.tolist(), "acf_bearing": acf_b.tolist(), "frames_per_s": 10}
    print(json.dumps({k: round(v, 3) for k, v in out.items() if not isinstance(v, list)}, indent=1))
    print("equivalent R inflation to account for correlation: range x%.1f, bearing x%.1f"
          % (1 / out["neff_ratio_range"], 1 / out["neff_ratio_bearing"]))
    with open(os.path.join(ROOT, "results", "data", "measurement_correlation.json"), "w") as f:
        json.dump(out, f, indent=2)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    lags = np.arange(maxlag + 1) / 10
    ax.plot(lags, acf_r, "o-", label="range error")
    ax.plot(lags, acf_b, "s-", label="bearing error")
    ax.axhline(0, c="k", lw=0.8)
    ax.set(xlabel="lag [s] (camera at 10 Hz)", ylabel="autocorrelation", title="Camera measurement error is not white")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "meas_autocorr.png"), dpi=130)


if __name__ == "__main__":
    main()
