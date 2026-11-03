"""Part E: repeated randomised trials under nominal and disturbed conditions, plus
the R-scaling (estimator consistency vs responsiveness) and LQR-weight
(control effort vs tracking error) trade-off studies.

Every trial randomises the start pose and obstacle layout (seeded, so the
results are reproducible). Results go to results/data/experiments.csv and one
representative log per condition to results/data/logs/.

Usage: python scripts/run_experiments.py [--trials 15] [--jobs 2] [--only nominal,push]
"""
import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tb3nav.controller import LQRParams
from tb3nav.ekf import EKFParams
from tb3nav.sim import TrialConfig, run_trial

ROOT = os.path.join(os.path.dirname(__file__), "..")


def conditions(n):
    C = {}
    C["nominal"] = (TrialConfig(), n)
    C["push_8N"] = (TrialConfig(push_force=8.0, push_time=12.0), n)
    nr = TrialConfig(push_force=8.0, push_time=12.0); nr.ekf = EKFParams(lockout_frames=0)
    C["push_8N_no_recovery"] = (nr, n)
    C["wheel_radius_+3%"] = (TrialConfig(wheel_radius_err=0.03), n)
    C["low_friction_0.3"] = (TrialConfig(floor_friction=0.3), n)
    C["frame_drop_50%"] = (TrialConfig(occlusion_prob=0.5), n)
    C["blind_8s"] = (TrialConfig(occlusion_window=(8.0, 16.0)), n)
    C["image_noise_blur"] = (TrialConfig(pixel_noise=10.0, blur=1.0), n)
    C["dim_light_40%"] = (TrialConfig(light_scale=0.4), n)
    C["combined"] = (TrialConfig(push_force=8.0, push_time=12.0, wheel_radius_err=0.03, floor_friction=0.6,
                                 occlusion_prob=0.5, pixel_noise=10.0, blur=1.0), n)
    odo = TrialConfig(); odo.ekf = EKFParams(use_camera=False)
    C["odometry_only"] = (odo, max(n // 2, 5))
    # R sweep (k multiplies the characterised R) on a disturbed condition so that
    # responsiveness matters: 8 N push + 3 % wheel radius error
    for rs in (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
        c = TrialConfig(wheel_radius_err=0.03, push_force=8.0, push_time=12.0)
        c.ekf = EKFParams(r_scale=rs)
        C[f"Rx{rs:g}"] = (c, max(n // 2, 5))
    # LQR cross-track weight sweep (effort vs tracking), 8 N push
    for ql in (0.25, 2.5, 25.0, 250.0, 2500.0):
        c = TrialConfig(push_force=8.0, push_time=12.0)
        c.lqr = LQRParams(q_lat=ql)
        C[f"q_lat{ql:g}"] = (c, max(n // 2, 5))
    return C


def _job(args):
    name, cfg, seed, keep = args
    cfg = deepcopy(cfg)
    cfg.seed = seed
    try:
        log = run_trial(cfg)
    except Exception as e:  # a crashed trial is a failed trial, recorded as such
        return name, seed, {"status": f"error: {e}", "success": False}, None
    return name, seed, log["metrics"], (log if keep else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--only", type=str, default="")
    a = ap.parse_args()
    C = conditions(a.trials)
    if a.only:
        C = {k: v for k, v in C.items() if k in a.only.split(",")}
    jobs = []
    for name, (cfg, n) in C.items():
        for i in range(n):
            jobs.append((name, cfg, 1000 + i, i == 0))
    os.makedirs(os.path.join(ROOT, "results", "data", "logs"), exist_ok=True)
    out_csv = os.path.join(ROOT, "results", "data", "experiments.csv")
    ckpt = os.path.join(ROOT, "results", "data", "experiments.jsonl")   # one line per finished trial
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                r = json.loads(line)
                done[(r["condition"], r["seed"])] = r
    todo = [j for j in jobs if (j[0], j[2]) not in done]
    print(f"{len(jobs) - len(todo)} trials already done, {len(todo)} to run", flush=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.jobs) as pool, open(ckpt, "a") as fck:
        for k, (name, seed, met, log) in enumerate(pool.imap_unordered(_job, todo)):
            row = {"condition": name, "seed": seed, **met}
            done[(name, seed)] = row
            fck.write(json.dumps(row) + "\n"); fck.flush()
            if log is not None:
                np.save(os.path.join(ROOT, "results", "data", "logs", f"{name}.npy"), log, allow_pickle=True)
            print(f"[{k+1}/{len(todo)}] {name:18s} seed {seed} {met.get('status')} "
                  f"dock {met.get('dock_pos_err_m', float('nan'))*100:.1f} cm", flush=True)
    write_csv(done, out_csv)


def write_csv(done, out_csv):
    rows = sorted(done.values(), key=lambda r: (r["condition"], r["seed"]))
    keys = sorted({k for r in rows for k in r.keys()}, key=lambda k: (k not in ("condition", "seed"), k))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved", os.path.abspath(out_csv))


if __name__ == "__main__":
    main()
