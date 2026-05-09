"""
Verify the full scene (robot + table + YCB objects) loads and runs.

Reports the combined model size, lists all bodies and joints, settles physics
for a moment, and opens the viewer.

Usage:
    python scripts/verify_full_scene.py
    python scripts/verify_full_scene.py --headless
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="Seconds of physics to settle before reporting.")
    args = ap.parse_args()

    print(f"Loading {SCENE_XML} ...")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    print(f"\nModel summary:")
    print(f"  nq    : {model.nq}    (generalized coords)")
    print(f"  nv    : {model.nv}    (DoF)")
    print(f"  nu    : {model.nu}    (actuators)")
    print(f"  nbody : {model.nbody}")
    print(f"  njnt  : {model.njnt}")
    print(f"  ngeom : {model.ngeom}")

    # Apply home keyframe (the robot's home pose)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf >= 0:
        data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
        data.ctrl[:] = model.key_ctrl[kf]
        # Free joints (objects) keep their default initial qpos from the MJCF
        mujoco.mj_forward(model, data)

    # Settle physics
    print(f"\nSettling for {args.settle}s of physics...")
    n_steps = int(args.settle / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    # Report robot tool position
    flange_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flange")
    if flange_id >= 0:
        flange_world = data.site_xpos[flange_id]
        print(f"\nRobot flange world position: {flange_world}")

    # Report YCB object positions
    print(f"\nYCB object world positions after settling:")
    for body_name in ["banana", "mug", "cracker_box", "mustard_bottle", "tennis_ball"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            x, y, z = data.xpos[bid]
            print(f"  {body_name:18s} ({x:+.3f}, {y:+.3f}, {z:+.3f})")
        else:
            print(f"  {body_name}: NOT FOUND")

    # Check for active contacts at this point
    print(f"\nActive contacts after settling: {data.ncon}")

    if args.headless:
        return

    print(f"\nOpening viewer. Close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
