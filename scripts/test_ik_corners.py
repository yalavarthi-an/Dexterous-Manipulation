"""
IK validation test: move the Piper arm's flange to the 4 corners of the
table and the center, verifying the arm can reach the full workspace.

Each target is 10cm above the table surface. The script:
  1. Solves position-only IK for each target
  2. Animates the arm moving there (PD controller tracking IK solution)
  3. Reports the final position error
  4. Opens a viewer showing the full sequence

This serves as a pre-flight check before attempting grasps: if the arm
can't reach a table corner, no grasp at that location will work either.

Usage:
    python scripts/test_ik_corners.py
    python scripts/test_ik_corners.py --headless
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

from src.planning.ik_solver import solve_ik_position, N_ARM_JOINTS

SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"

# Table geometry (must match full_scene.xml)
TABLE_CENTER_X = 0.5
TABLE_HALF_X = 0.30
TABLE_HALF_Y = 0.40
TABLE_Z = 0.70
HOVER_HEIGHT = 0.10  # fly 10cm above the table surface

# 5 table corner/center targets, all 10cm above the table
TABLE_TARGETS = {
    "center":     (TABLE_CENTER_X, 0.0, TABLE_Z + HOVER_HEIGHT),
    "near-left":  (TABLE_CENTER_X - TABLE_HALF_X + 0.05, TABLE_HALF_Y - 0.05, TABLE_Z + HOVER_HEIGHT),
    "near-right": (TABLE_CENTER_X - TABLE_HALF_X + 0.05, -TABLE_HALF_Y + 0.05, TABLE_Z + HOVER_HEIGHT),
    "far-left":   (TABLE_CENTER_X + TABLE_HALF_X - 0.05, TABLE_HALF_Y - 0.05, TABLE_Z + HOVER_HEIGHT),
    "far-right":  (TABLE_CENTER_X + TABLE_HALF_X - 0.05, -TABLE_HALF_Y + 0.05, TABLE_Z + HOVER_HEIGHT),
}

# Actual YCB object positions (10cm above each for a top-down approach)
OBJECT_TARGETS = {
    "banana":         (0.40, -0.20, TABLE_Z + HOVER_HEIGHT),
    "mug":            (0.55,  0.18, TABLE_Z + HOVER_HEIGHT),
    "cracker_box":    (0.40,  0.20, TABLE_Z + HOVER_HEIGHT),
    "mustard_bottle": (0.60, -0.18, TABLE_Z + HOVER_HEIGHT),
    "tennis_ball":    (0.65,  0.05, TABLE_Z + HOVER_HEIGHT),
}

# Combined: test everything
TARGETS = {**TABLE_TARGETS, **OBJECT_TARGETS}


def move_to_target(model, data, q_target, duration=2.0, viewer=None):
    """Smoothly move the arm to q_target over `duration` seconds using
    linear interpolation of joint angles — no jerky jumps."""
    q_start = data.qpos[:N_ARM_JOINTS].copy()
    base_ctrl = data.ctrl.copy()
    steps = int(duration / model.opt.timestep)

    for i in range(steps):
        # Linear interpolation: t goes from 0 to 1 over the duration
        t = (i + 1) / steps
        # Smooth easing (cubic ease-in-out for natural acceleration)
        if t < 0.5:
            t_smooth = 4 * t * t * t
        else:
            t_smooth = 1 - (-2 * t + 2) ** 3 / 2
        q_interp = q_start + t_smooth * (q_target - q_start)
        base_ctrl[:N_ARM_JOINTS] = q_interp
        data.ctrl[:] = base_ctrl

        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)


def get_flange_pos(model, data):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flange")
    return data.site_xpos[sid].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    print(f"Loading {SCENE_XML} ...")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # Apply home keyframe
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf >= 0:
        data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
        data.ctrl[:] = model.key_ctrl[kf]
    mujoco.mj_forward(model, data)

    # Settle
    for _ in range(200):
        mujoco.mj_step(model, data)

    # Record home position
    home_q = data.qpos[:N_ARM_JOINTS].copy()
    home_ctrl = data.ctrl.copy()

    print(f"\nFlange at home: {get_flange_pos(model, data)}")
    print(f"Table surface z={TABLE_Z:.2f}, targets at z={TABLE_Z + HOVER_HEIGHT:.2f}")
    print()

    # ---- Solve IK for all targets first (offline, no sim disturbance) ----
    results = {}
    print("Solving IK for each target (multi-start with 5 restarts)...")
    print(f"  {'target':18s}  {'world pos':>30s}  {'IK':>4s}  {'error (mm)':>10s}")
    print("  " + "-" * 70)

    # Table corners first
    print("  --- Table corners ---")
    for name, pos in TABLE_TARGETS.items():
        q_sol, err = solve_ik_position(model, data, np.array(pos))
        status = "OK" if q_sol is not None else "FAIL"
        err_mm = err * 1000
        print(f"  {name:18s}  ({pos[0]:.2f}, {pos[1]:+.2f}, {pos[2]:.2f})  {status:>4s}  {err_mm:8.1f} mm")
        results[name] = {"pos": np.array(pos), "q": q_sol, "err": err}

    # Object positions
    print("  --- Object positions (10cm above each) ---")
    for name, pos in OBJECT_TARGETS.items():
        q_sol, err = solve_ik_position(model, data, np.array(pos))
        status = "OK" if q_sol is not None else "FAIL"
        err_mm = err * 1000
        print(f"  {name:18s}  ({pos[0]:.2f}, {pos[1]:+.2f}, {pos[2]:.2f})  {status:>4s}  {err_mm:8.1f} mm")
        results[name] = {"pos": np.array(pos), "q": q_sol, "err": err}

    # Count successes
    n_ok = sum(1 for r in results.values() if r["q"] is not None)
    n_total = len(results)
    n_obj_ok = sum(1 for name in OBJECT_TARGETS if results.get(name, {}).get("q") is not None)
    print(f"\n  IK success: {n_ok}/{n_total} total, {n_obj_ok}/{len(OBJECT_TARGETS)} objects")

    if n_ok == 0:
        print("  No IK solutions found — arm cannot reach any target.")
        return

    # ---- Animate the arm visiting each reachable OBJECT position ----
    def run_animation(viewer):
        for name in OBJECT_TARGETS:
            result = results.get(name)
            if result is None or result["q"] is None:
                print(f"\n  Skipping {name} (IK failed)")
                continue

            print(f"\n  Moving to {name}...")

            # Smooth motion to IK solution
            move_to_target(model, data, result["q"], duration=2.5, viewer=viewer)

            # Settle briefly
            for _ in range(int(0.3 / model.opt.timestep)):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(model.opt.timestep)

            # Measure actual flange position
            mujoco.mj_forward(model, data)
            actual = get_flange_pos(model, data)
            err = np.linalg.norm(actual - result["pos"]) * 1000
            print(f"    target:  ({result['pos'][0]:.3f}, {result['pos'][1]:+.3f}, {result['pos'][2]:.3f})")
            print(f"    actual:  ({actual[0]:.3f}, {actual[1]:+.3f}, {actual[2]:.3f})")
            print(f"    error:   {err:.1f} mm")

        # Return to home
        print(f"\n  Returning to home...")
        move_to_target(model, data, home_q, duration=2.0, viewer=viewer)

        print("\n  Done! All targets visited.")
        print("  Close the viewer window to exit." if viewer else "")

    if args.headless:
        run_animation(viewer=None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_animation(viewer)
            # Keep viewer open
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()