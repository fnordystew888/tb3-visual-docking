"""LiDAR occupancy grid + A* global planner.

Only the room walls and the dock are known in advance. Obstacles are unknown and
are discovered online: every LiDAR scan is projected into the world with the EKF
pose estimate (never ground truth) and accumulated as log-odds hits. The grid is
inflated by the robot radius plus a safety margin and A* replans whenever the
current path crosses newly occupied cells.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation

from . import scene as S

RES = 0.05                       # m / cell
INFLATE = S.BASE_RADIUS + 0.08   # m


class OccupancyGrid:
    def __init__(self):
        self.x0, self.y0 = -S.ROOM_X, -S.ROOM_Y
        self.nx, self.ny = int(round(2 * S.ROOM_X / RES)), int(round(2 * S.ROOM_Y / RES))
        self.logodds = np.zeros((self.nx, self.ny))
        self.known = np.zeros((self.nx, self.ny), bool)
        self.known[0, :] = self.known[-1, :] = self.known[:, 0] = self.known[:, -1] = True
        # dock station box is part of the known map
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.cell_centre(i, j)
                if x > S.DOCK_FACE_X and abs(y) < 0.15:
                    self.known[i, j] = True
        self._inflated = None
        r = int(np.ceil(INFLATE / RES))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        self.kernel = (xx ** 2 + yy ** 2) <= (INFLATE / RES) ** 2

    def cell(self, x, y):
        return int((x - self.x0) / RES), int((y - self.y0) / RES)

    def cell_centre(self, i, j):
        return self.x0 + (i + 0.5) * RES, self.y0 + (j + 0.5) * RES

    def integrate_scan(self, pose, scan, hit=0.85, miss=-0.4):
        """Ray-cast each beam from the estimated pose; free cells along the ray, hit at the end."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        for a, r in zip(scan.angles, scan.ranges):
            ang = pose[2] + a
            rr = min(r, S.LIDAR_RANGE) if np.isfinite(r) else S.LIDAR_RANGE
            n = int(rr / RES)
            for k in range(1, n):
                i, j = self.cell(pose[0] + k * RES * np.cos(ang), pose[1] + k * RES * np.sin(ang))
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    self.logodds[i, j] = max(self.logodds[i, j] + miss, -4.0)
            if np.isfinite(r):
                i, j = self.cell(pose[0] + r * np.cos(ang), pose[1] + r * np.sin(ang))
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    self.logodds[i, j] = min(self.logodds[i, j] + hit, 6.0)
        self._inflated = None

    def occupied(self):
        return self.known | (self.logodds > 1.5)

    def inflated(self):
        if self._inflated is None:
            self._inflated = binary_dilation(self.occupied(), structure=self.kernel)
        return self._inflated

    def path_blocked(self, path, start_idx=0):
        inf = self.inflated()
        for p in path[start_idx:]:
            i, j = self.cell(*p[:2])
            if 0 <= i < self.nx and 0 <= j < self.ny and inf[i, j]:
                return True
        return False


def astar(grid: OccupancyGrid, start, goal):
    """8-connected A* on the inflated grid; returns list of (x, y) or None."""
    inf = grid.inflated().copy()
    si, sj = grid.cell(*start)
    gi, gj = grid.cell(*goal)
    # the robot's own cell and the goal must be free even if the inflation touches them
    for (ci, cj) in ((si, sj), (gi, gj)):
        rr = int(np.ceil(INFLATE / RES))
        inf[max(ci - 1, 0):ci + 2, max(cj - 1, 0):cj + 2] = False
    if not (0 <= si < grid.nx and 0 <= sj < grid.ny):
        return None
    moves = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    openq = [(0.0, 0.0, (si, sj))]
    came, cost = {}, {(si, sj): 0.0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == (gi, gj):
            break
        if g > cost.get(cur, np.inf):
            continue
        for di, dj, c in moves:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < grid.nx and 0 <= nb[1] < grid.ny) or inf[nb]:
                continue
            ng = g + c
            if ng < cost.get(nb, np.inf):
                cost[nb] = ng
                came[nb] = cur
                heapq.heappush(openq, (ng + np.hypot(nb[0] - gi, nb[1] - gj), ng, nb))
    if (gi, gj) not in cost:
        return None
    cells = [(gi, gj)]
    while cells[-1] != (si, sj):
        cells.append(came[cells[-1]])
    cells.reverse()
    pts = [grid.cell_centre(i, j) for i, j in cells]
    pts[0], pts[-1] = tuple(start[:2]), tuple(goal[:2])
    return shortcut(grid, pts, inf)


def _free_line(grid, inf, a, b):
    n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / (RES / 2)), 1)
    for t in np.linspace(0, 1, n + 1):
        i, j = grid.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if inf[i, j]:
            return False
    return True


def shortcut(grid, pts, inf):
    """Greedy line-of-sight shortcutting of the A* cell path."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _free_line(grid, inf, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def densify(waypoints, ds=0.02, corner_radius=0.25):
    """Turn a polyline into a smooth reference path with heading and curvature.
    Corners are rounded with circular arcs (radius <= corner_radius) so that the
    reference curvature is bounded and the robot does not need to stop and turn."""
    W = [np.array(w, float) for w in waypoints]
    if len(W) < 2:
        return np.array([[W[0][0], W[0][1], 0.0, 0.0]])
    pts = [W[0]]
    for k in range(1, len(W) - 1):
        a, b, c = W[k - 1], W[k], W[k + 1]
        u1, u2 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        turn = np.arctan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2)
        if abs(turn) < 1e-3:
            pts.append(b)
            continue
        d = min(corner_radius * np.tan(abs(turn) / 2), 0.45 * np.linalg.norm(b - a), 0.45 * np.linalg.norm(c - b))
        R = d / np.tan(abs(turn) / 2)
        p_in, p_out = b - u1 * d, b + u2 * d
        n = np.array([-u1[1], u1[0]]) * np.sign(turn)
        centre = p_in + n * R
        a0 = np.arctan2(p_in[1] - centre[1], p_in[0] - centre[0])
        m = max(int(abs(turn) * R / ds), 2)
        for t in np.linspace(0, turn, m):
            pts.append(centre + R * np.array([np.cos(a0 + t), np.sin(a0 + t)]))
    pts.append(W[-1])
    # resample uniformly
    P = np.array(pts)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.hstack([[True], seg > 1e-6])
    P = P[keep]
    s = np.hstack([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    ss = np.arange(0, s[-1], ds)
    ss = np.append(ss, s[-1])
    x, y = np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])
    th = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    kappa = np.gradient(th) / np.maximum(np.gradient(ss), 1e-6)
    kappa = np.convolve(kappa, np.ones(5) / 5, mode="same")
    return np.column_stack([x, y, th, kappa, ss])
