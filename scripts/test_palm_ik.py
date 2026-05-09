"""
Step 4.2: Verify the palm_center site and test 6-DOF IK targeting it.

Tests:
  1. Print palm_center world position + orientation at home pose
  2. Solve 6-DOF IK to place the palm at known poses:
     a. Palm-down above table center (top-down grasp orientation)
     b. Palm facing +X at table center height (side grasp orientation)
  3. Animate the arm to each pose and measure position+orientation error
  4. Print the palm's world-frame axes at each pose for visual verification

Usage:
    python scripts/test_palm_ik.py
    python scripts/test_palm_ik.py --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.planning.ik_solver import (
    solve_ik_position, solve_ik_pose, N_ARM_JOINTS,
    _get_site_pose,
)

SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"


def print_site_info(model, data, site_name):
    """Print world position and orientation of a site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site_xpos[sid]
    rot = data.site_xmat[sid].reshape(3, 3)
    print(f"  {site_name} position:  ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
    print(f"  {site_name} +X (world): ({rot[0,0]:+.3f}, {rot[1,0]:+.3f}, {rot[2,0]:+.3f})")
    print(f"  {site_name} +Y (world): ({rot[0,1]:+.3f}, {rot[1,1]:+.3f}, {rot[2,1]:+.3f})")
    print(f"  {site_name} +Z (world): ({rot[0,2]:+.3f}, {rot[1,2]:+.3f}, {rot[2,2]:+.3f})")
    return pos.copy(), rot.copy()


def smooth_move(model, data, q_target, duration=3.0, viewer=None):
    """Cubic-eased joint interpolation."""
    q_start = data.qpos[:N_ARM_JOINTS].copy()
    base_ctrl = data.ctrl.copy()
    steps = int(duration / model.opt.timestep)
    for i in range(steps):
        t = (i + 1) / steps
        if t < 0.5:
            t_smooth = 4 * t * t * t
        else:
            t_smooth = 1 - (-2 * t + 2) ** 3 / 2
        base_ctrl[:N_ARM_JOINTS] = q_start + t_smooth * (q_target - q_start)
        data.ctrl[:] = base_ctrl
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # Apply home and settle
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
    data.ctrl[:] = model.key_ctrl[kf]
    for _ in range(300):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    home_ctrl = data.ctrl.copy()

    # ================================================================
    print("=" * 70)
    print("PART 1: Palm site position at home pose")
    print("=" * 70)

    print("\n  Flange site (link6 origin):")
    flange_pos, flange_rot = print_site_info(model, data, "flange")

    print("\n  Palm center site (grasp target):")
    palm_pos, palm_rot = print_site_info(model, data, "palm_center")

    offset = palm_pos - flange_pos
    print(f"\n  Flange → Palm offset: ({offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f})")
    print(f"  Offset magnitude: {np.linalg.norm(offset)*1000:.1f} mm")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 2: 6-DOF IK targeting palm_center")
    print("=" * 70)

    # Test pose A: palm-down above table center (top-down grasp)
    # Palm +Z should point DOWN (world -Z) — this is the finger direction
    # (In the palm frame, -Z is the finger direction. The palm_center site
    #  inherits the palm body's frame, so site +Z = palm +Z = wrist direction,
    #  and site -Z = palm -Z = finger direction.)
    # Wait — I need to check what the palm frame actually looks like at home.
    # From Part 1 output, I'll know the palm +Z direction.
    
    # For a top-down grasp, we want the fingers (palm -Z) pointing DOWN (world -Z).
    # So palm +Z should point UP (world +Z).
    # palm +X can be world +X (forward)
    # palm +Y = cross(+Z, +X) = cross(world+Z, world+X) = world -Y
    R_topdown = np.array([
        [1.0,  0.0,  0.0],   # palm +X = world +X
        [0.0, -1.0,  0.0],   # palm +Y = world -Y
        [0.0,  0.0, -1.0],   # palm +Z = world -Z... wait
    ])
    # Hmm, I need to be more careful. Let me think about what orientation we want.
    # At home pose, I'll see what the palm's natural frame looks like.
    # Then for top-down, I want fingers pointing down = palm_-Z pointing world_-Z
    # = palm_+Z pointing world_+Z
    # For a right-handed frame: X cross Y = Z
    # palm +X = world +X, palm +Y = world -Y → Z = X×Y = (+X)×(-Y) = +Z ✓
    # Actually (+1,0,0)×(0,-1,0) = (0*0-0*(-1), 0*0-1*0, 1*(-1)-0*0) = (0,0,-1)
    # That gives palm +Z = world -Z, which means fingers (palm -Z) = world +Z (pointing UP)
    # That's wrong! Let me redo this.
    
    # For top-down grasp: fingers point DOWN = world -Z
    # fingers = palm -Z, so palm +Z = world +Z
    # We need palm +Z = world +Z
    # Choose: palm +X = world +X, palm +Y = world +Y
    # Check: X×Y = (+X)×(+Y) = +Z ✓ → right-handed, palm +Z = world +Z ✓
    R_topdown = np.eye(3)  # identity = palm frame aligned with world frame
    # palm +Z = world +Z (up), palm -Z = world -Z (down = finger direction) ✓

    target_pos_A = np.array([0.45, 0.0, 0.85])  # above table center
    print(f"\n  Test A: Top-down grasp pose")
    print(f"    Target pos: {target_pos_A}")
    print(f"    Target rot: palm +Z = world +Z (fingers point down)")

    q_A, pos_err_A, rot_err_A = solve_ik_pose(
        model, data, target_pos_A, R_topdown,
        site_name="palm_center", n_restarts=5)

    if q_A is not None:
        print(f"    IK OK: pos_err={pos_err_A*1000:.1f}mm, rot_err={np.degrees(rot_err_A):.1f}°")
    else:
        print(f"    IK FAILED: pos_err={pos_err_A*1000:.1f}mm, rot_err={np.degrees(rot_err_A):.1f}°")

    # Test pose B: side grasp facing +X (fingers point toward +X)
    # fingers = palm -Z pointing world +X → palm +Z = world -X
    # palm +X = world -Z (palm-out faces down)
    # palm +Y = cross(+Z, +X) = cross(world_-X, world_-Z)
    #         = (-1,0,0)×(0,0,-1) = (0*(-1)-0*0, 0*(-1)-(-1)*(-1), (-1)*0-0*(-1))
    #         = (0, -1, 0) = world -Y
    # So: palm +X = (0,0,-1), +Y = (0,-1,0), +Z = (-1,0,0)
    R_side = np.array([
        [ 0.0,  0.0, -1.0],   # col0: palm +X in world
        [ 0.0, -1.0,  0.0],   # col1: palm +Y in world
        [-1.0,  0.0,  0.0],   # col2: palm +Z in world
    ])
    # Verify right-handed: det(R) should be +1
    det = np.linalg.det(R_side)

    target_pos_B = np.array([0.40, -0.10, 0.80])  # beside mustard area
    print(f"\n  Test B: Side grasp pose (fingers → +X)")
    print(f"    Target pos: {target_pos_B}")
    print(f"    Target rot: palm +Z = world -X (fingers point +X), det={det:.0f}")

    q_B, pos_err_B, rot_err_B = solve_ik_pose(
        model, data, target_pos_B, R_side,
        site_name="palm_center", n_restarts=5)

    if q_B is not None:
        print(f"    IK OK: pos_err={pos_err_B*1000:.1f}mm, rot_err={np.degrees(rot_err_B):.1f}°")
    else:
        print(f"    IK FAILED: pos_err={pos_err_B*1000:.1f}mm, rot_err={np.degrees(rot_err_B):.1f}°")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 3: Animate and verify poses visually")
    print("=" * 70)

    def run_visual(viewer):
        if q_A is not None:
            print("\n  Moving to top-down pose (Test A)...")
            smooth_move(model, data, q_A, duration=3.0, viewer=viewer)
            # Settle
            for _ in range(int(1.0 / model.opt.timestep)):
                mujoco.mj_step(model, data)
                if viewer: viewer.sync(); time.sleep(model.opt.timestep)
            mujoco.mj_forward(model, data)
            print("  Palm center at top-down pose:")
            print_site_info(model, data, "palm_center")

        if q_B is not None:
            print("\n  Moving to side-grasp pose (Test B)...")
            smooth_move(model, data, q_B, duration=3.0, viewer=viewer)
            for _ in range(int(1.0 / model.opt.timestep)):
                mujoco.mj_step(model, data)
                if viewer: viewer.sync(); time.sleep(model.opt.timestep)
            mujoco.mj_forward(model, data)
            print("  Palm center at side-grasp pose:")
            print_site_info(model, data, "palm_center")

        # Return to home
        print("\n  Returning to home...")
        smooth_move(model, data, model.key_qpos[kf, :N_ARM_JOINTS],
                    duration=2.0, viewer=viewer)

    if args.headless:
        run_visual(None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_visual(viewer)
            print("\n  Close viewer to exit.")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
