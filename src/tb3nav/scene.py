"""MuJoCo scene for Track A: TurtleBot3 Burger in a 5 m x 4 m room with obstacles,
ArUco landmarks on the walls and an ArUco-marked charging (dock) station.

The scene XML is generated in Python so that obstacle positions can be
randomised per trial. All geometry follows the TurtleBot3 Burger datasheet where
it matters for the model (wheel radius 0.033 m, wheel separation 0.160 m,
max linear speed 0.22 m/s, LDS-01 LiDAR 360 deg / 3.5 m, 5 Hz).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")

# ---------------------------------------------------------------- robot constants
WHEEL_RADIUS = 0.033          # m   (TB3 Burger datasheet)
WHEEL_SEP = 0.160             # m   (TB3 Burger datasheet)
BASE_RADIUS = 0.09            # m   (footprint used for collision / inflation)
V_MAX = 0.22                  # m/s (TB3 Burger datasheet)
W_MAX = 2.84                  # rad/s (TB3 Burger datasheet)
LIDAR_HEIGHT = 0.17           # m
LIDAR_BEAMS = 90              # 4 deg resolution (LDS-01 is 1 deg; 90 keeps sim fast)
LIDAR_RANGE = 3.5             # m
CAM_POS_BASE = np.array([0.07, 0.0, 0.15])   # camera position in base frame [m]
CAM_W, CAM_H, CAM_FOVY = 640, 480, 62.0       # Raspberry Pi Camera v2-like

# ---------------------------------------------------------------- room / landmarks
ROOM_X, ROOM_Y = 2.5, 2.0     # half extents [m]
MARKER_Z = 0.15               # marker centre height = camera height

# id: (x, y, facing yaw [rad] = direction of marker normal, side length [m])
LANDMARKS = {
    1: (-ROOM_X + 0.005, 1.0, 0.0, 0.20),
    2: (-ROOM_X + 0.005, -1.0, 0.0, 0.20),
    3: (-1.2, ROOM_Y - 0.005, -np.pi / 2, 0.20),
    4: (-1.2, -ROOM_Y + 0.005, np.pi / 2, 0.20),
    5: (0.3, ROOM_Y - 0.005, -np.pi / 2, 0.20),
    6: (0.3, -ROOM_Y + 0.005, np.pi / 2, 0.20),
    7: (1.6, ROOM_Y - 0.005, -np.pi / 2, 0.20),
    8: (1.6, -ROOM_Y + 0.005, np.pi / 2, 0.20),
}
DOCK_ID = 0
DOCK_FACE_X = 2.40            # x of the front face of the charging station
DOCK_MARKER_SIZE = 0.10
LANDMARKS[DOCK_ID] = (DOCK_FACE_X - 0.001, 0.0, np.pi, DOCK_MARKER_SIZE)
DOCK_POSE = np.array([2.20, 0.0, 0.0])        # target base pose when docked (x, y, yaw)
PRE_DOCK = np.array([1.55, 0.0, 0.0])         # start of the straight docking approach


def marker_png(marker_id: int) -> str:
    """Write the ArUco (DICT_4X4_50) texture with a white quiet zone; return path."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    path = os.path.join(ASSET_DIR, f"aruco_{marker_id}.png")
    if not os.path.exists(path):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        img = cv2.aruco.generateImageMarker(d, marker_id, 240)       # 6 modules x 40 px
        img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return path


def marker_black_side(marker_id: int) -> float:
    """The texture is 320 px with a 40 px white border, so the black square (which
    is what OpenCV's corner detector returns) is 240/320 of the textured plane."""
    return LANDMARKS[marker_id][3] * 240.0 / 320.0


@dataclass
class SceneConfig:
    obstacles: list = field(default_factory=list)  # list of (type, x, y, size)
    floor_friction: float = 1.0                     # applied to floor AND tyres (MuJoCo takes the max of the pair)
    wheel_radius_true: float = WHEEL_RADIUS        # lets us inject model mismatch
    light_scale: float = 1.0


def random_obstacles(rng: np.random.Generator, n: int = 3) -> list:
    """Place n obstacles in the middle of the room, away from start, the docking
    corridor and each other."""
    obs = []
    tries = 0
    if n > 0:   # first obstacle always blocks the straight line start -> dock
        obs.append(("box" if rng.random() < 0.5 else "cylinder", float(rng.uniform(-0.6, 0.6)),
                    float(rng.uniform(-0.25, 0.25)), float(rng.uniform(0.15, 0.22))))
    while len(obs) < n and tries < 1000:
        tries += 1
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-1.4, 1.4)
        size = rng.uniform(0.12, 0.20)
        kind = "box" if rng.random() < 0.5 else "cylinder"
        if any(np.hypot(x - ox, y - oy) < 0.75 for _, ox, oy, _ in obs):
            continue
        obs.append((kind, float(x), float(y), float(size)))
    return obs


def build_xml(cfg: SceneConfig) -> str:
    assets, geoms = [], []
    for mid, (mx, my, myaw, side) in LANDMARKS.items():
        p = marker_png(mid)
        assets.append(f'<texture name="tex{mid}" type="2d" file="{p}"/>'
                      f'<material name="mat{mid}" texture="tex{mid}" texrepeat="1 1" texuniform="false"/>')
        # plane normal (+z) must point along the marker facing direction:
        # rotate about y by -90 deg (normal -> -x... we want +x) then about z.
        # euler (xyz intrinsic): first z-yaw then y.
        yaw_deg = np.degrees(myaw)
        geoms.append(
            f'<body name="marker{mid}" pos="{mx} {my} {MARKER_Z}" euler="0 0 {yaw_deg}">'
            f'<geom type="plane" size="{side/2} {side/2} 0.01" material="mat{mid}" '
            f'euler="0 90 0" contype="0" conaffinity="0" group="2"/></body>')

    for i, (kind, x, y, s) in enumerate(cfg.obstacles):
        if kind == "box":
            geoms.append(f'<geom name="obs{i}" type="box" size="{s} {s} 0.25" pos="{x} {y} 0.25" rgba="0.55 0.35 0.2 1"/>')
        else:
            geoms.append(f'<geom name="obs{i}" type="cylinder" size="{s} 0.25" pos="{x} {y} 0.25" rgba="0.3 0.45 0.6 1"/>')

    lidar_sites = "\n".join(
        f'<site name="lidar{k}" pos="0 0 {LIDAR_HEIGHT}" '
        f'zaxis="{np.cos(2*np.pi*k/LIDAR_BEAMS):.6f} {np.sin(2*np.pi*k/LIDAR_BEAMS):.6f} 0" size="0.005" group="4"/>'
        for k in range(LIDAR_BEAMS))
    lidar_sensors = "\n".join(f'<rangefinder name="r{k}" site="lidar{k}" cutoff="{LIDAR_RANGE}"/>'
                              for k in range(LIDAR_BEAMS))

    r = cfg.wheel_radius_true
    ls = cfg.light_scale
    wall_t = 0.05
    xml = f"""
<mujoco model="tb3_track_a">
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="960"/><headlight ambient="{0.35*ls} {0.35*ls} {0.35*ls}" diffuse="{0.5*ls} {0.5*ls} {0.5*ls}"/>
    <quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.74 0.74 0.72" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="10 8"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="{0.6*ls} {0.6*ls} {0.6*ls}"/>
    <geom name="floor" type="plane" size="{ROOM_X+0.5} {ROOM_Y+0.5} 0.05" material="floor" friction="{cfg.floor_friction} 0.005 0.0001"/>
    <geom name="wall_e" type="box" size="{wall_t} {ROOM_Y} 0.3" pos="{ROOM_X+wall_t} 0 0.3" rgba="0.92 0.92 0.9 1"/>
    <geom name="wall_w" type="box" size="{wall_t} {ROOM_Y} 0.3" pos="{-ROOM_X-wall_t} 0 0.3" rgba="0.92 0.92 0.9 1"/>
    <geom name="wall_n" type="box" size="{ROOM_X+2*wall_t} {wall_t} 0.3" pos="0 {ROOM_Y+wall_t} 0.3" rgba="0.92 0.92 0.9 1"/>
    <geom name="wall_s" type="box" size="{ROOM_X+2*wall_t} {wall_t} 0.3" pos="0 {-ROOM_Y-wall_t} 0.3" rgba="0.92 0.92 0.9 1"/>
    <geom name="dock" type="box" size="{(ROOM_X-DOCK_FACE_X)/2} 0.15 0.12" pos="{(ROOM_X+DOCK_FACE_X)/2} 0 0.12" rgba="0.15 0.6 0.25 1"/>
    {''.join(geoms)}

    <body name="base" pos="0 0 {r + 0.001}">
      <freejoint name="root"/>
      <inertial pos="-0.02 0 0.05" mass="1.0" diaginertia="0.008 0.008 0.010"/>
      <geom name="chassis" type="box" size="0.07 0.07 0.04" pos="-0.02 0 0.05" rgba="0.2 0.2 0.2 1" mass="0"/>
      <geom name="top" type="cylinder" size="0.035 0.02" pos="-0.02 0 {LIDAR_HEIGHT-0.02}" rgba="0.1 0.1 0.1 1" mass="0" contype="0" conaffinity="0"/>
      <geom name="camhousing" type="box" size="0.01 0.02 0.01" pos="{CAM_POS_BASE[0]-0.012} 0 {CAM_POS_BASE[2]}" rgba="0.8 0.1 0.1 1" mass="0" contype="0" conaffinity="0"/>
      <geom name="caster_f" type="sphere" size="0.01" pos="0.05 0 {0.01 - r}" condim="1" priority="1" friction="0 0 0" mass="0"/>
      <geom name="caster_r" type="sphere" size="0.01" pos="-0.08 0 {0.01 - r}" condim="1" priority="1" friction="0 0 0" mass="0"/>
      <camera name="rgb" pos="{CAM_POS_BASE[0]} {CAM_POS_BASE[1]} {CAM_POS_BASE[2]}" xyaxes="0 -1 0 0 0 1" fovy="{CAM_FOVY}"/>
      {lidar_sites}
      <body name="wheel_l" pos="0 {WHEEL_SEP/2} 0">
        <joint name="wl" type="hinge" axis="0 1 0" damping="0.002" armature="0.005"/>
        <geom type="cylinder" size="{r} 0.009" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="0.05" friction="{cfg.floor_friction} 0.005 0.0001"/>
      </body>
      <body name="wheel_r" pos="0 {-WHEEL_SEP/2} 0">
        <joint name="wr" type="hinge" axis="0 1 0" damping="0.002" armature="0.005"/>
        <geom type="cylinder" size="{r} 0.009" euler="90 0 0" rgba="0.1 0.1 0.1 1" mass="0.05" friction="{cfg.floor_friction} 0.005 0.0001"/>
      </body>
    </body>
    <camera name="top" pos="0 0 6.2" xyaxes="1 0 0 0 1 0" fovy="45"/>
  </worldbody>
  <actuator>
    <velocity name="ml" joint="wl" kv="0.6" ctrlrange="{-V_MAX/WHEEL_RADIUS} {V_MAX/WHEEL_RADIUS}" forcerange="-1.4 1.4"/>
    <velocity name="mr" joint="wr" kv="0.6" ctrlrange="{-V_MAX/WHEEL_RADIUS} {V_MAX/WHEEL_RADIUS}" forcerange="-1.4 1.4"/>
  </actuator>
  <sensor>
    <jointpos name="enc_l" joint="wl"/>
    <jointpos name="enc_r" joint="wr"/>
    {lidar_sensors}
  </sensor>
</mujoco>"""
    return xml
