"""
Per-joint verification: tests each Piper arm joint individually.

For each of the 6 arm joints:
  1. Commands it to mid-range while holding others at home
  2. Waits for the PD controller to converge (5 seconds)
  3. Measures the tracking error (commanded vs actual angle)
  4. Commands it back to home

Also tests the full IK pipeline with extended settle time.

Usage:
    python scripts/verify_joints_tracking.py
    python scripts/verify_joints_tracking.py --headless
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # Apply home
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
    data.ctrl[:] = model.key_ctrl[kf]
    mujoco.mj_forward(model, data)

    # Settle to home
    for _ in range(500):
        mujoco.mj_step(model, data)
    home_ctrl = data.ctrl.copy()
    home_qpos = data.qpos[:N_ARM_JOINTS].copy()

    print("=" * 70)
    print("PART 1: Per-joint tracking verification")
    print("=" * 70)
    print(f"\n  {'joint':8s}  {'range':>20s}  {'home':>6s}  {'target':>6s}  {'actual':>6s}  {'error':>8s}  {'status'}")
    print("  " + "-" * 75)

    for jid in range(N_ARM_JOINTS):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = model.jnt_range[jid]
        home_val = home_qpos[jid]

        # Target: midpoint of the range
        target = (lo + hi) / 2.0

        # Command this joint to target, others stay at home
        ctrl = home_ctrl.copy()
        ctrl[jid] = target
        data.ctrl[:] = ctrl

        # Run 5 seconds of physics
        for _ in range(int(5.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)

        actual = data.qpos[jid]
        err_deg = np.degrees(abs(actual - target))
        status = "OK" if err_deg < 5.0 else "DRIFT" if err_deg < 15.0 else "FAIL"

        print(f"  {jname:8s}  [{np.degrees(lo):+7.1f}, {np.degrees(hi):+7.1f}]°"
              f"  {np.degrees(home_val):+6.1f}°"
              f"  {np.degrees(target):+6.1f}°"
              f"  {np.degrees(actual):+6.1f}°"
              f"  {err_deg:6.2f}°"
              f"  {status}")

        # Return to home
        data.ctrl[:] = home_ctrl
        for _ in range(int(3.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 2: IK tracking with extended settle time")
    print("=" * 70)

    # Reset to home
    data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
    data.ctrl[:] = model.key_ctrl[kf]
    for _ in range(500):
        mujoco.mj_step(model, data)

    targets = {
        "banana":     (0.40, -0.20, 0.80),
        "mug":        (0.55,  0.18, 0.80),
        "cracker_box":(0.40,  0.20, 0.80),
        "mustard":    (0.60, -0.18, 0.80),
    }

    settle_times = [1.0, 3.0, 5.0]

    flange_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "flange")

    print(f"\n  {'target':12s}", end="")
    for st in settle_times:
        print(f"  {'settle ' + str(st) + 's':>14s}", end="")
    print(f"  {'IK err':>8s}")
    print("  " + "-" * 60)

    for name, pos in targets.items():
        pos_arr = np.array(pos)
        q_sol, ik_err = solve_ik_position(model, data, pos_arr)

        if q_sol is None:
            print(f"  {name:12s}  IK FAILED ({ik_err*1000:.0f}mm)")
            continue

        errors = []
        for st in settle_times:
            # Reset position AND velocity, then apply IK solution
            data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
            data.qvel[:] = 0.0  # critical: reset velocity between tests
            data.ctrl[:] = home_ctrl.copy()
            data.ctrl[:N_ARM_JOINTS] = q_sol

            # Run settle_time seconds of physics
            for _ in range(int(st / model.opt.timestep)):
                mujoco.mj_step(model, data)

            mujoco.mj_forward(model, data)
            actual_pos = data.site_xpos[flange_sid].copy()
            err_mm = np.linalg.norm(actual_pos - pos_arr) * 1000
            errors.append(err_mm)

        print(f"  {name:12s}", end="")
        for err_mm in errors:
            print(f"  {err_mm:12.1f}mm", end="")
        print(f"  {ik_err*1000:6.1f}mm")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("PART 3: Gravity compensation check")
    print("=" * 70)

    # Check if RUKA bodies have gravcomp
    ruka_bodies = ["ruka_palm", "ruka_index_mcp", "ruka_middle_mcp",
                   "ruka_ring_mcp", "ruka_pinky_mcp", "ruka_thumb_mcp"]
    print(f"\n  {'body':20s}  {'gravcomp':>8s}  {'mass (g)':>10s}")
    print("  " + "-" * 45)
    for bname in ruka_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            gc = model.body_gravcomp[bid]
            mass = model.body_mass[bid] * 1000  # kg to g
            status = "YES" if gc > 0.5 else "NO ← needs fix"
            print(f"  {bname:20s}  {status:>8s}  {mass:8.1f} g")

    # Total RUKA mass
    total_ruka = 0
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and bname.startswith("ruka_"):
            total_ruka += model.body_mass[bid]
    print(f"\n  Total RUKA mass: {total_ruka*1000:.1f} g")
    print(f"  This hangs off the wrist with NO gravity compensation.")
    print(f"  The wrist joints (kp=10) must fight {total_ruka * 9.81:.2f} N of uncompensated weight.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()