"""
Verify the tabletop scene loads, objects rest on the table without falling
off the edges or through the surface.

Usage:
    python scripts/verify_tabletop.py            # interactive viewer
    python scripts/verify_tabletop.py --headless # text only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "assets" / "scene" / "tabletop_only.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="Seconds of physics to let objects settle before reporting.")
    args = ap.parse_args()

    print(f"Loading {SCENE_XML} ...")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    print(f"\nModel summary:")
    print(f"  bodies: {model.nbody}, joints: {model.njnt} (free joints = movable objects)")
    free_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    ]
    print(f"  movable objects: {free_joints}")

    # Settle objects
    print(f"\nSimulating {args.settle}s for objects to settle...")
    n_steps = int(args.settle / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    print(f"\nFinal object positions after settling:")
    print(f"  {'object':18s}  {'x':>7s} {'y':>7s} {'z':>7s}    {'on_table':>8s}")
    table_top_z = 0.70
    table_x_min, table_x_max = 0.5 - 0.30, 0.5 + 0.30
    table_y_min, table_y_max = -0.40, 0.40

    bad = 0
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or name in ("world", "table"):
            continue
        x, y, z = data.xpos[i]
        on_table = (
            table_x_min - 0.05 <= x <= table_x_max + 0.05 and
            table_y_min - 0.05 <= y <= table_y_max + 0.05 and
            z >= table_top_z - 0.01
        )
        flag = "OK" if on_table else "OFF-TABLE"
        if not on_table:
            bad += 1
        print(f"  {name:18s}  {x:+.3f} {y:+.3f} {z:+.3f}    {flag}")

    if bad == 0:
        print("\n  ✓ All objects resting on the table.")
    else:
        print(f"\n  WARNING: {bad} object(s) off the table — check positions in build_objects_fragment.py")

    if args.headless:
        return

    print("\nOpening viewer. Close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()