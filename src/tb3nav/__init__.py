"""TurtleBot3 visual waypoint navigation and ArUco docking (41014 Track A)."""
import os
import sys

# Headless Linux needs a software/offscreen GL backend for MuJoCo's renderer.
# macOS and Windows use their native backend, so nothing is set there.
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
