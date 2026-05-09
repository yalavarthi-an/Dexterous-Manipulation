"""
Diagnostic: reports the world-frame position of the flange site, palm body,
and all 5 fingertip sites at the home pose.

Used to figure out the correct mount transform empirically: by comparing
the fingertip positions with the flange position, we can see which world
direction the fingers extend in, and design the mount to align that with
the desired tool-axis direction.

Usage:
    python scripts/diagnose_mount.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "assets" / "mounted" / "piper_ruka_scene.xml"


def get_site_pos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        return None
    return data.site_xpos[sid].copy()


def get_body_pos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    return data.xpos[bid].copy()


def get_body_xmat(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    return data.xmat[bid].reshape(3, 3).copy()


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # Apply home keyframe
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[kf_id]
    data.qvel[:] = 0.0
    data.ctrl[:] = model.key_ctrl[kf_id]
    mujoco.mj_forward(model, data)  # compute kinematics, no integration

    # Reference frames
    flange_pos = get_site_pos(model, data, "flange")
    palm_pos   = get_body_pos(model, data, "ruka_palm")
    palm_R     = get_body_xmat(model, data, "ruka_palm")
    link6_pos  = get_body_pos(model, data, "link6")
    link6_R    = get_body_xmat(model, data, "link6")

    print("=" * 70)
    print("World positions at home pose (axes: X=forward+right, Y=left, Z=up)")
    print("=" * 70)
    print(f"  link6 origin:     {link6_pos}")
    print(f"  flange site:      {flange_pos}")
    print(f"  ruka_palm origin: {palm_pos}")
    print(f"  palm offset from link6: {palm_pos - link6_pos}")
    print()
    print("link6 frame axes in world coords:")
    print(f"  link6 +X = {link6_R[:,0]}")
    print(f"  link6 +Y = {link6_R[:,1]}")
    print(f"  link6 +Z = {link6_R[:,2]}   ← this is where fingers SHOULD point")
    print()
    print("ruka_palm frame axes in world coords:")
    print(f"  palm  +X = {palm_R[:,0]}")
    print(f"  palm  +Y = {palm_R[:,1]}")
    print(f"  palm  +Z = {palm_R[:,2]}")
    print()
    print("Fingertip world positions and direction from palm:")
    for name in ["ruka_index_tip", "ruka_middle_tip", "ruka_ring_tip",
                 "ruka_pinky_tip", "ruka_thumb_tip"]:
        tip = get_site_pos(model, data, name)
        if tip is None:
            print(f"  {name}: not found")
            continue
        direction = tip - palm_pos
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            unit = direction / norm
        else:
            unit = direction
        print(f"  {name:20s} world={tip}  dir_unit={unit}  dist={norm:.3f}")
    print()
    print("Direction analysis: project finger direction onto link6 axes")
    avg_finger_world = np.zeros(3)
    n = 0
    for name in ["ruka_index_tip", "ruka_middle_tip", "ruka_ring_tip", "ruka_pinky_tip"]:
        tip = get_site_pos(model, data, name)
        if tip is not None:
            avg_finger_world += (tip - palm_pos)
            n += 1
    if n > 0:
        avg_finger_world /= n
        avg_finger_world /= (np.linalg.norm(avg_finger_world) + 1e-9)
        print(f"  Average finger direction (world): {avg_finger_world}")
        print(f"  Projected on link6 +X: {np.dot(avg_finger_world, link6_R[:,0]):+.3f}")
        print(f"  Projected on link6 +Y: {np.dot(avg_finger_world, link6_R[:,1]):+.3f}")
        print(f"  Projected on link6 +Z: {np.dot(avg_finger_world, link6_R[:,2]):+.3f}")
        print()
        print("  → Fingers currently point most strongly along link6's:")
        projs = {
            "+X": np.dot(avg_finger_world, link6_R[:,0]),
            "-X": -np.dot(avg_finger_world, link6_R[:,0]),
            "+Y": np.dot(avg_finger_world, link6_R[:,1]),
            "-Y": -np.dot(avg_finger_world, link6_R[:,1]),
            "+Z": np.dot(avg_finger_world, link6_R[:,2]),
            "-Z": -np.dot(avg_finger_world, link6_R[:,2]),
        }
        best = max(projs, key=projs.get)
        print(f"     {best} axis (with strength {projs[best]:.3f})")
        print()
        print("  We WANT them to point along link6's +Z.")
        print(f"  Therefore we need a rotation that maps {best} → +Z in link6's frame.")

    print("=" * 70)


if __name__ == "__main__":
    main()
