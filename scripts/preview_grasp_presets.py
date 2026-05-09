"""
Preview grasp presets: loads the robot in the full scene and lets you
cycle through each grasp preset to see the hand shape.

Usage:
    python scripts/preview_grasp_presets.py

Controls (type in the terminal, then press Enter):
    <Enter> or 'n'  : next preset
    'p'             : previous preset
    'o'             : reset to open hand
    'q'             : quit

The preset name and joint values are printed to the terminal each time you
switch. Adjust the YAML file and re-run to iterate on finger angles.
"""

from __future__ import annotations

import sys
import time
import select
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"
PRESETS_YAML = REPO_ROOT / "configs" / "grasp_presets.yaml"

# Joint names in actuator order (indices 6..20 of ctrl)
HAND_JOINT_NAMES = [
    "index_mcp",  "index_dip",  "index_pip",
    "middle_mcp", "middle_dip", "middle_pip",
    "ring_mcp",   "ring_dip",   "ring_pip",
    "pinky_mcp",  "pinky_dip",  "pinky_pip",
    "thumb_cmc",  "thumb_mcp",  "thumb_ip",
]

ARM_ACTUATOR_COUNT = 6  # first 6 actuators are Piper arm


def load_presets() -> dict[str, np.ndarray]:
    """Load grasp presets from YAML, return {name: 15-element array}."""
    with open(PRESETS_YAML) as f:
        raw = yaml.safe_load(f)

    presets = {}
    for name, joints in raw.items():
        if name == "object_grasps":
            continue  # skip the object mapping section
        if not isinstance(joints, dict):
            continue
        arr = np.array([joints[k] for k in HAND_JOINT_NAMES], dtype=np.float64)
        presets[name] = arr
    return presets


def apply_preset(data: mujoco.MjData, preset: np.ndarray, home_ctrl: np.ndarray):
    """Set the hand actuators to the preset values, keep arm at home."""
    ctrl = home_ctrl.copy()
    ctrl[ARM_ACTUATOR_COUNT:] = preset
    data.ctrl[:] = ctrl


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
    home_ctrl = data.ctrl.copy()

    # Settle
    for _ in range(200):
        mujoco.mj_step(model, data)

    presets = load_presets()
    preset_names = list(presets.keys())
    idx = 0

    print(f"\nLoaded {len(presets)} presets: {preset_names}")
    print("Controls: <Enter>=next, 'p'=prev, 'o'=open, 'q'=quit\n")

    def show_current():
        name = preset_names[idx]
        arr = presets[name]
        apply_preset(data, arr, home_ctrl)
        print(f"[{idx+1}/{len(presets)}] {name}")
        for jn, v in zip(HAND_JOINT_NAMES, arr):
            print(f"  {jn:14s} = {v:+.2f}")
        print()

    show_current()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            # Check for keyboard input from terminal
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                line = sys.stdin.readline().strip().lower()
                if line in ("", "n", "next"):
                    idx = (idx + 1) % len(preset_names)
                    show_current()
                elif line in ("p", "prev"):
                    idx = (idx - 1) % len(preset_names)
                    show_current()
                elif line == "o":
                    idx = preset_names.index("open")
                    show_current()
                elif line in ("q", "quit", "exit"):
                    break

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
