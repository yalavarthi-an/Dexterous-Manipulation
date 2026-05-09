"""
Test the IK solver against the grasp proposals from Task 3.

Loads outputs/grasp_proposals.json (saved by visualize_grasps.py --save),
then for each proposal:
  - Tries position-only IK to pre_grasp_pos
  - Tries position-only IK to palm_pos
  - Tries full pose IK to (palm_pos, palm_quat)
  - Prints residual errors

If position IK succeeds but pose IK fails, the issue is orientation
reachability — the position is reachable but with a wrong wrist orientation
for that hand to grasp from there. We can then iterate on grasp_proposal.py
to relax the orientation requirement or pick a different grasp axis.

Usage:
    python scripts/test_ik.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.planning.ik_solver import (
    solve_ik_position, solve_ik_pose, N_ARM_JOINTS
)

SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"
PROPOSALS_JSON = REPO_ROOT / "outputs" / "grasp_proposals.json"

PALM_SITE = "palm"


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert (w,x,y,z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def main():
    if not PROPOSALS_JSON.exists():
        print(f"ERROR: {PROPOSALS_JSON} not found.")
        print("Run: python scripts/visualize_grasps.py --save")
        sys.exit(1)

    print(f"Loading scene: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # Apply home keyframe so IK starts from a sensible state
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[kf]
    data.ctrl[:] = model.key_ctrl[kf]
    mujoco.mj_forward(model, data)

    # Load grasp proposals
    with open(PROPOSALS_JSON) as f:
        proposals = json.load(f)
    print(f"Loaded {len(proposals)} grasp proposals\n")

    # Verify the palm site exists
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)
    if sid < 0:
        print(f"ERROR: site '{PALM_SITE}' not found in model")
        sys.exit(1)
    print(f"Palm site at home pose: {data.site_xpos[sid]}\n")

    # Header
    print(f"{'#':>2}  {'object':14s}  {'type':16s}  {'app':>4s}  "
          f"{'pre_pos':>8s}  {'pos':>8s}  {'pose_p':>7s}  {'pose_r':>7s}  status")
    print("-" * 100)

    n_full_success = 0
    n_pos_success = 0

    for i, p in enumerate(proposals[:10]):  # top 10
        obj = p["object_name"]
        gtype = p["grasp_type"]
        approach = p["approach"]
        pre_pos = np.array(p["pre_grasp_pos"])
        palm_pos = np.array(p["palm_pos"])
        palm_quat = np.array(p["palm_quat"])
        palm_R = quat_to_matrix(palm_quat)

        # Test 1: position IK to pre-grasp
        q_pre, err_pre = solve_ik_position(
            model, data, pre_pos, site_name=PALM_SITE, n_restarts=5,
        )

        # Test 2: position IK to grasp position (using pre as seed)
        q_grasp, err_grasp = solve_ik_position(
            model, data, palm_pos, site_name=PALM_SITE,
            q_init=q_pre if q_pre is not None else None,
            n_restarts=5,
        )

        # Test 3: full pose IK to (grasp_pos, grasp_rot)
        # Use relaxed rot_tol — for grasping, 15° orientation error is
        # easily compensated by finger curl, no need for 5° precision.
        q_pose, err_pose_p, err_pose_r = solve_ik_pose(
            model, data, palm_pos, palm_R, site_name=PALM_SITE,
            q_init=q_grasp if q_grasp is not None else None,
            n_restarts=10,
            rot_tol=0.26,  # ~15 degrees
            pos_tol=0.01,  # 1 cm position tolerance
        )

        # Format errors as mm and degrees
        s_pre = f"{err_pre*1000:7.1f}mm" if q_pre is not None else "    FAIL"
        s_grasp = f"{err_grasp*1000:7.1f}mm" if q_grasp is not None else "    FAIL"
        s_pose_p = f"{err_pose_p*1000:6.1f}mm" if q_pose is not None else "   FAIL"
        s_pose_r = f"{np.degrees(err_pose_r):5.1f}°" if q_pose is not None else "  FAIL"

        # Status assessment
        if q_pose is not None:
            status = "✅ FULL POSE OK"
            n_full_success += 1
            n_pos_success += 1
        elif q_grasp is not None:
            status = "⚠️  pos OK, pose FAIL (orientation problem)"
            n_pos_success += 1
        elif q_pre is not None:
            status = "⚠️  pre-grasp OK, grasp FAIL"
        else:
            status = "❌ unreachable"

        print(f"{i+1:>2}  {obj:14s}  {gtype:16s}  {approach:>4s}  "
              f"{s_pre}  {s_grasp}  {s_pose_p}  {s_pose_r}  {status}")

    print("-" * 100)
    print(f"\nSummary:")
    print(f"  Full pose IK success: {n_full_success}/{len(proposals[:10])}")
    print(f"  Position IK success:  {n_pos_success}/{len(proposals[:10])}")

    if n_full_success == 0 and n_pos_success > 0:
        print("\n  All position-only IKs work but full-pose IKs fail.")
        print("  → The palm orientations in grasp_proposal.py are likely off.")
        print("  → Investigate the R_palm matrix conventions.")
    elif n_full_success < len(proposals[:10]):
        print(f"\n  Some grasps unreachable. Common causes:")
        print(f"  - Object too far from base (e.g., far corner)")
        print(f"  - Side-approach palm position pushes hand outside workspace")
        print(f"  - Orientation requires elbow-flip past joint limits")


if __name__ == "__main__":
    main()