"""
Comprehensive mount verification for the Piper + RUKA assembly.

Checks the mount transform at three levels of rigor:
  1. SYMBOLIC: parse the quaternion in piper_ruka.xml and compute its rotation matrix.
     Compare each axis (X, Y, Z) of the rotated palm frame to the desired alignment.
  2. NUMERICAL: load the model, run forward kinematics at home, and verify world-frame
     positions and orientations of the palm relative to link6.
  3. INTERPENETRATION: list all body-pair penetrations near the mount (palm <-> link*).

Usage:
    python scripts/verify_mount.py
"""

from pathlib import Path
import re

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = REPO_ROOT / "assets" / "mounted" / "piper_ruka.xml"
SCENE_XML = REPO_ROOT / "assets" / "mounted" / "piper_ruka_scene.xml"


# ===========================================================================
# Quaternion math (MuJoCo convention: q = (w, x, y, z))
# ===========================================================================
def quat_to_matrix(q):
    """Convert (w,x,y,z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def angle_between(v1, v2):
    """Return angle in degrees between two unit vectors."""
    c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    return np.degrees(np.arccos(c))


# ===========================================================================
# Level 1: Symbolic verification (just the quaternion)
# ===========================================================================
def parse_mount_from_xml(xml_path):
    text = xml_path.read_text()
    m = re.search(
        r'<body name="ruka_palm" pos="([^"]+)" quat="([^"]+)"',
        text,
    )
    if not m:
        raise RuntimeError("Could not find ruka_palm mount line.")
    pos = np.array([float(x) for x in m.group(1).split()])
    quat = np.array([float(x) for x in m.group(2).split()])
    return pos, quat


def symbolic_check(quat):
    print("=" * 72)
    print("LEVEL 1: SYMBOLIC VERIFICATION (pure math from the quaternion)")
    print("=" * 72)
    print(f"  quaternion (w,x,y,z): ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
    R = quat_to_matrix(quat)
    print(f"  rotation matrix R (palm-frame axes expressed in link6 frame):")
    for row in R:
        print(f"    [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")
    print()

    # The columns of R are the palm-frame basis vectors expressed in link6 frame.
    palm_X_in_link6 = R[:, 0]  # palm +X in link6 coords
    palm_Y_in_link6 = R[:, 1]  # palm +Y in link6 coords
    palm_Z_in_link6 = R[:, 2]  # palm +Z in link6 coords

    # Recall RUKA's native conventions:
    #   palm +X = thumb side
    #   palm +Y = palm-out (back of hand opposite this)
    #   palm -Z = fingers extend forward
    finger_dir_in_link6 = -palm_Z_in_link6  # because fingers are in palm's -Z
    palm_out_in_link6 = palm_Y_in_link6
    thumb_side_in_link6 = palm_X_in_link6

    print("  Where each meaningful palm direction now points in link6 frame:")
    print(f"    Finger direction (was palm -Z): {finger_dir_in_link6}")
    print(f"    Palm-out         (was palm +Y): {palm_out_in_link6}")
    print(f"    Thumb side       (was palm +X): {thumb_side_in_link6}")
    print()

    # Desired alignment: fingers along link6 +Z (the tool axis)
    link6_X = np.array([1, 0, 0])
    link6_Y = np.array([0, 1, 0])
    link6_Z = np.array([0, 0, 1])

    finger_to_toolaxis_angle = angle_between(finger_dir_in_link6, link6_Z)
    print(f"  Angle between finger direction and link6 +Z (tool axis): {finger_to_toolaxis_angle:.3f}°")
    if finger_to_toolaxis_angle < 1.0:
        print("    PASS: Fingers are aligned with the tool axis (within 1°).")
    else:
        print(f"    FAIL: Misalignment exceeds 1° threshold.")
    print()

    # Now classify what direction the palm faces
    print("  Palm-out direction analysis (which link6 axis is most aligned?):")
    for label, ax in [("+X", link6_X), ("-X", -link6_X), ("+Y", link6_Y),
                       ("-Y", -link6_Y), ("+Z", link6_Z), ("-Z", -link6_Z)]:
        a = angle_between(palm_out_in_link6, ax)
        print(f"    palm-out vs link6 {label}: {a:6.2f}°")

    return R


# ===========================================================================
# Level 2: Numerical verification (run the model, measure)
# ===========================================================================
def numerical_check(xml_path):
    print()
    print("=" * 72)
    print("LEVEL 2: NUMERICAL VERIFICATION (forward kinematics at home pose)")
    print("=" * 72)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[kf]
    data.ctrl[:] = model.key_ctrl[kf]
    mujoco.mj_forward(model, data)

    # World-frame rotations of link6 and ruka_palm
    bid_l6 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    bid_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ruka_palm")
    R_l6 = data.xmat[bid_l6].reshape(3, 3)
    R_palm = data.xmat[bid_palm].reshape(3, 3)
    pos_l6 = data.xpos[bid_l6]
    pos_palm = data.xpos[bid_palm]

    # Compute relative rotation: R_link6_to_palm = R_link6^T @ R_palm
    R_rel = R_l6.T @ R_palm
    print(f"  Relative rotation (link6 -> palm) measured from kinematics:")
    for row in R_rel:
        print(f"    [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f}]")
    print()

    # Relative position
    rel_pos = R_l6.T @ (pos_palm - pos_l6)
    print(f"  Relative position (palm origin in link6 frame): {rel_pos}")
    print(f"    Expected: pos in MJCF should match.")
    print()

    # Now compute fingertip positions and compare to link6 +Z direction
    print("  Per-finger direction analysis (in world frame):")
    finger_world_avg = np.zeros(3)
    n = 0
    for tip_name in ["ruka_index_tip", "ruka_middle_tip",
                     "ruka_ring_tip", "ruka_pinky_tip"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
        if sid < 0:
            continue
        tip = data.site_xpos[sid]
        d = tip - pos_palm
        d_unit = d / np.linalg.norm(d)
        finger_world_avg += d_unit
        n += 1
    if n > 0:
        finger_world_avg /= n
        finger_world_avg /= np.linalg.norm(finger_world_avg)
        link6_z_world = R_l6[:, 2]
        ang = angle_between(finger_world_avg, link6_z_world)
        print(f"    Average finger direction (world):  {finger_world_avg}")
        print(f"    link6 +Z direction        (world): {link6_z_world}")
        print(f"    Angular deviation: {ang:.3f}°")
        if ang < 10:
            print(f"    PASS: Finger axis aligns with tool axis (within 10° including splay).")
        else:
            print(f"    FAIL: Misalignment exceeds 10°.")


# ===========================================================================
# Level 3: Interpenetration check
# ===========================================================================
def interpenetration_check(xml_path):
    print()
    print("=" * 72)
    print("LEVEL 3: INTERPENETRATION CHECK")
    print("=" * 72)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[kf]
    data.ctrl[:] = model.key_ctrl[kf]
    for _ in range(20):
        mujoco.mj_step(model, data)

    print(f"  Active contacts at home: {data.ncon}")
    if data.ncon == 0:
        print("    PASS: No contact pairs at home pose.")
        return

    bad = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -1e-4:  # negative = penetration
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
            print(f"    PENETRATION: {b1} <-> {b2}: dist={c.dist:+.4f}m")
            bad += 1
    if bad == 0:
        print("    PASS: No significant penetrations.")
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    pos, quat = parse_mount_from_xml(ROBOT_XML)
    print(f"Mount transform from XML: pos={pos}, quat={quat}\n")
    symbolic_check(quat)
    numerical_check(SCENE_XML)
    interpenetration_check(SCENE_XML)
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
