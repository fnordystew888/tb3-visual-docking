"""End-to-end trial: LiDAR mapping + A* planning, EKF (encoders + camera), LQR
tracking and visual docking, in MuJoCo.

Ground truth from the simulator is read in exactly two places, both marked
`# GROUND TRUTH (evaluation only)`: logging the true pose for metrics, and
placing the robot at its random start. Nothing the controller, planner or EKF
consumes is derived from it.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

from . import model as Mdl
from . import scene as S
from .controller import LQRParams, LQRTracker
from .ekf import EKF, EKFParams
from .planner import OccupancyGrid, astar, densify
from .sensors import Camera, Encoders, Lidar, base_pose

PHYS_DT = 0.002
CTRL_EVERY = 25      # 20 Hz control + EKF predict
CAM_EVERY = 50       # 10 Hz camera
LIDAR_EVERY = 100    # 5 Hz LiDAR
START_NOMINAL = np.array([-1.85, 0.0, 0.0])   # start pad: the only prior the robot has
P0 = np.diag([0.15 ** 2, 0.30 ** 2, 0.30 ** 2])


@dataclass
class TrialConfig:
    seed: int = 0
    n_obstacles: int = 3
    # disturbances / variations
    push_force: float = 0.0          # [N] lateral push on the base
    push_time: float = 20.0          # [s]
    push_duration: float = 0.3       # [s]
    wheel_radius_err: float = 0.0    # true radius = nominal * (1 + err); EKF keeps nominal
    floor_friction: float = 1.0
    occlusion_prob: float = 0.0      # probability a camera frame is lost
    occlusion_window: tuple = None   # (t0, t1): camera blind in this window
    pixel_noise: float = 0.0         # [grey levels] image noise std
    blur: float = 0.0                # Gaussian blur sigma [px]
    encoder_noise: float = 0.0       # extra random-walk encoder noise [rad/read]
    light_scale: float = 1.0
    ekf: EKFParams = field(default_factory=EKFParams)
    lqr: LQRParams = field(default_factory=LQRParams)
    timeout: float = 150.0
    dock_speed: float = 0.08
    record_video: bool = False
    video_every: int = 2             # record every n-th camera frame


def run_trial(cfg: TrialConfig, verbose=False, frame_cb=None):
    rng = np.random.default_rng(cfg.seed)
    obstacles = S.random_obstacles(rng, cfg.n_obstacles)
    scfg = S.SceneConfig(obstacles=obstacles, floor_friction=cfg.floor_friction,
                         wheel_radius_true=S.WHEEL_RADIUS * (1 + cfg.wheel_radius_err),
                         light_scale=cfg.light_scale)
    m = mujoco.MjModel.from_xml_string(S.build_xml(scfg))
    d = mujoco.MjData(m)

    # GROUND TRUTH (evaluation only): random placement around the start pad
    start = START_NOMINAL + np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
    d.qpos[0:3] = [start[0], start[1], scfg.wheel_radius_true + 0.0005]
    d.qpos[3:7] = [np.cos(start[2] / 2), 0, 0, np.sin(start[2] / 2)]
    mujoco.mj_forward(m, d)

    enc = Encoders(m, cfg.encoder_noise, rng)
    lidar = Lidar(m, rng=rng)
    cam = Camera(m, pixel_noise=cfg.pixel_noise, blur=cfg.blur, rng=rng)
    ekf = EKF(START_NOMINAL.copy(), P0.copy(), cfg.ekf)
    grid = OccupancyGrid()
    ctrl = LQRTracker(cfg.lqr)
    base_vref = cfg.lqr.v_ref

    def plan(x_hat):
        wps = astar(grid, x_hat[:2], S.PRE_DOCK[:2])
        if wps is None:
            return None
        path = densify(list(wps) + [tuple(S.DOCK_POSE[:2])])
        return path

    path = plan(ekf.x)
    ctrl.set_path(path)
    predock_s = None

    def predock_index(P):
        return int(np.argmin(np.hypot(P[:, 0] - S.PRE_DOCK[0], P[:, 1] - S.PRE_DOCK[1])))

    pd_idx = predock_index(path)
    prev_enc = enc.read(d)
    log = {k: [] for k in ("t", "true", "est", "P", "ref_e", "v", "w", "n_obs", "nis", "path_idx", "ct_true")}
    robot_geoms = {m.geom("chassis").id}
    obstacle_geoms = {m.geom(f"obs{i}").id for i in range(len(obstacles))}
    collisions = 0
    replans, n_frames, frames = 0, 0, []
    obs_last = []
    v = w = 0.0
    status = "timeout"
    blocked_since = None
    front_clear = np.inf
    backup_until = -1.0
    n_backups = 0
    t_wall = time.time()
    step = 0
    while d.time < cfg.timeout:
        # ---------------- disturbance
        if cfg.push_force > 0 and cfg.push_time <= d.time < cfg.push_time + cfg.push_duration:
            yaw = base_pose(d)[2]   # direction only, to push sideways relative to the robot
            d.xfrc_applied[m.body("base").id, :3] = cfg.push_force * np.array([-np.sin(yaw), np.cos(yaw), 0])
        else:
            d.xfrc_applied[m.body("base").id, :3] = 0

        # ---------------- LiDAR -> map -> replan (5 Hz)
        if step % LIDAR_EVERY == 0:
            scan = lidar.read(d)
            grid.integrate_scan(ekf.x, scan)
            # LiDAR bumper: clearance in the +-40 deg forward cone (laser frame = base frame)
            fwd = (np.abs(Mdl.wrap(scan.angles)) < np.radians(40))
            front_clear = float(np.min(scan.ranges[fwd])) if np.any(np.isfinite(scan.ranges[fwd])) else np.inf
            if ctrl.idx < pd_idx and grid.path_blocked(ctrl.path[:pd_idx], ctrl.idx):
                new = plan(ekf.x)
                if new is not None:
                    ctrl.set_path(new)
                    pd_idx = predock_index(new)
                    replans += 1

        # ---------------- camera -> EKF update (10 Hz)
        if step % CAM_EVERY == 0:
            blind = (cfg.occlusion_window is not None and cfg.occlusion_window[0] <= d.time < cfg.occlusion_window[1])
            img = cam.render(d)
            obs = cam.detect(img)[0]
            if blind or rng.random() < cfg.occlusion_prob:
                obs = []
            ekf.update(obs)
            obs_last = obs
            if frame_cb is not None and n_frames % cfg.video_every == 0:
                frame_cb(d, img, obs, ekf, ctrl, grid, base_pose(d))
            n_frames += 1

        # ---------------- encoders -> EKF predict, LQR (20 Hz)
        if step % CTRL_EVERY == 0:
            e_now = enc.read(d)
            dphi = e_now - prev_enc
            prev_enc = e_now
            ekf.predict(dphi[0], dphi[1])
            ctrl.p.v_ref = cfg.dock_speed if ctrl.idx >= pd_idx - 10 else base_vref
            rem = ctrl.remaining()
            if rem < 0.01:
                v, w = 0.0, 0.0
                status = "docked"
            else:
                v, w, e = ctrl.step(ekf.x)
                # bumper: never drive forward into something < 12 cm from the chassis
                # edge; back off for 1 s and force a replan from the new position.
                # Disabled on the docking segment, where the dock itself is meant to be close.
                if ctrl.idx < pd_idx - 10 and front_clear < S.BASE_RADIUS + 0.12 and d.time > backup_until:
                    backup_until = d.time + 1.0
                    n_backups += 1
                    new = plan(ekf.x)
                    if new is not None:
                        ctrl.set_path(new); pd_idx = predock_index(new); replans += 1
                if d.time < backup_until:
                    v, w = -0.05, 0.0
            wl, wr = Mdl.body_to_wheels(v, w)
            d.ctrl[:] = [wl, wr]
            # GROUND TRUTH (evaluation only)
            gt = base_pose(d)
            log["t"].append(d.time); log["true"].append(gt); log["est"].append(ekf.x.copy())
            log["P"].append(ekf.P.copy()); log["ref_e"].append(ctrl.last_e.copy())
            log["v"].append(v); log["w"].append(w); log["n_obs"].append(len(obs_last))
            log["nis"].append(list(ekf.last_nis)); log["path_idx"].append(ctrl.idx)
            P_ = ctrl.path
            log["ct_true"].append(float(np.min(np.hypot(P_[:, 0] - gt[0], P_[:, 1] - gt[1]))))
            for c in d.contact[:d.ncon]:
                if (c.geom1 in obstacle_geoms) or (c.geom2 in obstacle_geoms):
                    collisions += 1
                    break
            if status == "docked":
                break
        mujoco.mj_step(m, d)
        step += 1

    log["collisions"] = collisions
    for k in ("t", "true", "est", "P", "ref_e", "v", "w", "n_obs", "path_idx", "ct_true"):
        log[k] = np.array(log[k])
    log["path"] = ctrl.path
    log["obstacles"] = obstacles
    log["grid"] = grid.occupied()
    log["start"] = start
    log["status"] = status
    log["replans"] = replans
    log["backups"] = n_backups
    log["ekf_updates"], log["ekf_rejects"], log["ekf_resets"] = ekf.n_upd, ekf.n_rej, ekf.n_resets
    log["wall_time"] = time.time() - t_wall
    log["metrics"] = metrics(log)
    if verbose:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in log["metrics"].items()})
    return log


def metrics(log):
    """All metrics use the simulator ground truth, which is only available here."""
    T, E, P = log["true"], log["est"], log["P"]
    err = E - T
    err[:, 2] = Mdl.wrap(err[:, 2])
    pos_err = np.hypot(err[:, 0], err[:, 1])
    nees = np.array([e @ np.linalg.solve(p, e) for e, p in zip(err, P)])
    # distance of the TRUE robot from the path that was active at that instant
    xt = log["ct_true"]
    fin = T[-1]
    dock_pos_err = float(np.hypot(fin[0] - S.DOCK_POSE[0], fin[1] - S.DOCK_POSE[1]))
    dock_yaw_err = float(abs(Mdl.wrap(fin[2] - S.DOCK_POSE[2])))
    success = log["status"] == "docked" and dock_pos_err < 0.05 and dock_yaw_err < np.radians(10)
    dt = np.diff(log["t"], prepend=log["t"][0])
    t = log["t"]
    conv = t >= 3.0     # after the initial fix from the start-pad prior
    e2 = log["ref_e"][:, 1]
    return {
        "status": log["status"],
        "est_rmse_pos_after3s_m": float(np.sqrt(np.mean(pos_err[conv] ** 2))),
        "track_rmse_after3s_m": float(np.sqrt(np.mean(np.asarray(xt)[conv] ** 2))),
        "ctrl_ct_rmse_m": float(np.sqrt(np.mean(e2[conv] ** 2))),   # cross-track error w.r.t. the ESTIMATE (what the LQR sees)
        "success": bool(success),
        "time_s": float(log["t"][-1]),
        "dock_pos_err_m": dock_pos_err,
        "dock_lat_err_m": float(abs(fin[1] - S.DOCK_POSE[1])),
        "dock_yaw_err_deg": float(np.degrees(dock_yaw_err)),
        "est_rmse_pos_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "est_max_pos_m": float(pos_err.max()),
        "est_rmse_yaw_deg": float(np.degrees(np.sqrt(np.mean(err[:, 2] ** 2)))),
        "final_est_err_m": float(pos_err[-1]),
        "nees_mean": float(nees.mean()),
        "nees_in_95": float(np.mean(nees < 7.81)),     # chi2 3 dof 95 %
        "track_rmse_m": float(np.sqrt(np.mean(xt ** 2))),
        "track_max_m": float(xt.max()),
        "effort": float(np.sum((np.array(log["v"]) ** 2 + 0.1 * np.array(log["w"]) ** 2) * dt)),
        "replans": log["replans"],
        "backups": log.get("backups", 0),
        "ekf_updates": log["ekf_updates"],
        "ekf_rejects": log["ekf_rejects"],
        "ekf_resets": log.get("ekf_resets", 0),
        "collision_steps": int(log["collisions"]),
    }
