"""
Quick camera preview: launches the viewer for full_scene.xml.
Use the keyboard shortcut '[' and ']' inside the viewer to cycle through
the available cameras (wrist_cam, scene_cam, plus the default free camera).

Usage:
    python scripts/preview_cameras.py
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"


def main():
    print(f"Loading {SCENE_XML} ...")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # Apply home keyframe
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf >= 0:
        data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
        data.ctrl[:] = model.key_ctrl[kf]
        mujoco.mj_forward(model, data)

    # Settle objects briefly
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    # List cameras
    print(f"\nAvailable cameras:")
    for cid in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        print(f"  [{cid}] {name}")
    print(f"  [-1] free camera (default)")
    print()
    print("Inside the viewer, press '[' and ']' to cycle through cameras.")
    print("Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
