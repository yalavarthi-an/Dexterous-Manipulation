"""
Grasp execution: multi-waypoint state machine.

Phases:
  HOME
    -> PRE-GRASP        : palm 15cm above grasp, palm horizontal facing down
    -> APPROACH-INT     : palm reoriented to grasp orientation, offset
                          5cm from final grasp along the approach axis
    -> APPROACH-FINAL   : move along approach axis to final grasp position
    -> CLOSE FINGERS    : slow close (2s) so object is not pushed away
    -> LIFT             : palm rises to a clear height above the table

Pre-grasp orientation is ALWAYS palm-horizontal (facing the table). This
separates "navigate over the object" from "align with the grasp" — making
side grasps less aggressive and reducing the risk of knocking objects.

The approach axis is computed from grasp_pos - obj_center for side grasps
(horizontal direction from object to palm). For top grasps, the approach
axis is world +Z (descend straight down).
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import mujoco
import numpy as np

from src.grasping.grasp_proposal import GraspProposal
from src.planning.ik_solver import solve_ik_pose, N_ARM_JOINTS


# ============================================================================
# Constants
# ============================================================================

N_HAND_JOINTS = 15
PALM_SITE = "palm"

# Waypoint geometry
PRE_GRASP_HEIGHT = 0.15      # palm 15 cm above grasp position (in world Z)
INTERMEDIATE_OFFSET = 0.05   # 5 cm before final grasp along the approach axis

# Lift target
LIFT_DELTA = 0.15            # palm rises 15 cm in world Z after grasp

# Phase durations (seconds)
DUR_HOME_TO_PREGRASP = 2.0          # long motion across workspace
DUR_PREGRASP_TO_INTERMEDIATE = 1.5  # may include big reorientation (side grasps)
DUR_INTERMEDIATE_TO_GRASP = 1.2     # short, careful approach
DUR_CLOSE_FINGERS = 2.0             # slow finger close so object is not pushed
DUR_GRASP_TO_LIFT = 1.5
DUR_PHASE_SETTLE = 0.3              # short settle between motions
DUR_FINAL_SETTLE = 1.0              # longer settle after lift

# IK pose tolerances (relaxed for grasping)
IK_POS_TOL = 0.01    # 1 cm
IK_ROT_TOL = 0.26    # ~15 degrees

# Success threshold
LIFT_SUCCESS_HEIGHT = 0.05    # 5 cm above initial Z = success
KNOCK_DETECTION_DROP = 0.05   # object dropped > this = knocked

# Mug-specific: insert-and-grip approach
MUG_SEMI_CLOSE_DURATION = 1.0    # time to semi-close at pre-grasp

# Function A: Insert — middle + ring fingers at 0.4, index + pinky fully curled
MUG_INSERT_FINGERS = np.array([
    2.2, 2.0, 2.0,   # index: FULLY CURLED (out of the way)
    0.4, 0.1, 0.1,   # middle: insert finger
    0.4, 0.1, 0.1,   # ring: insert finger
    2.2, 2.0, 2.0,   # pinky: FULLY CURLED (out of the way)
    0.0, 0.1, 0.1,   # thumb: NOT opposed, barely flexed
])

# Function B: Grip — middle + ring + thumb close, index + pinky stay curled
MUG_GRIP_FINGERS = np.array([
    2.2, 2.0, 2.0,   # index: stays curled
    1.5, 1.5, 1.5,   # middle: fully closed
    1.5, 1.5, 1.5,   # ring: fully closed
    2.2, 2.0, 2.0,   # pinky: stays curled
   -1.5, 1.0, 1.0,   # thumb: OPPOSED + closed
])

# Pre-grasp orientation: palm faces down, fingers extend forward
# Quaternion = 90 deg rotation around world +Y
#     R = [[0,0,1],[0,1,0],[-1,0,0]]
#     link6 +X (palm-out) -> world -Z   (palm down)
#     link6 +Y (thumb)    -> world +Y
#     link6 +Z (fingers)  -> world +X   (forward)
PALM_HORIZONTAL_QUAT = np.array([0.7071068, 0.0, 0.7071068, 0.0])


# ============================================================================
# Result dataclass
# ============================================================================

@dataclasses.dataclass
class TrajectoryResult:
    """Outcome of one execute_grasp() call."""
    object_name: str
    grasp_type: str
    approach: str
    success: bool
    failure_mode: Optional[str]   # None | IK_FAIL_PREGRASP | IK_FAIL_INTERMEDIATE
                                  # | IK_FAIL_GRASP | IK_FAIL_LIFT
                                  # | OBJECT_KNOCKED | GRASP_SLIP

    # Per-waypoint IK errors (-1.0 if that waypoint wasn't reached)
    pre_grasp_ik_pos_err: float
    pre_grasp_ik_rot_err: float
    intermediate_ik_pos_err: float
    intermediate_ik_rot_err: float
    grasp_ik_pos_err: float
    grasp_ik_rot_err: float
    lift_ik_pos_err: float
    lift_ik_rot_err: float

    # Object kinematics
    initial_object_z: float
    final_object_z: float
    lift_height: float

    # Per-timestep logs (for diagnostics / plotting)
    time_log: np.ndarray
    qpos_log: np.ndarray
    object_pos_log: np.ndarray


# ============================================================================
# Helpers
# ============================================================================

def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _smooth_motion(model, data, arm_start, arm_target, hand_target,
                   duration, log_fn=None, viewer=None):
    """Cubic ease-in/ease-out interpolation of arm joint targets in joint
    space. Hand control is held at `hand_target` throughout."""
    dt = model.opt.timestep
    n_steps = max(1, int(duration / dt))
    arm_start = np.asarray(arm_start, dtype=np.float64)
    arm_target = np.asarray(arm_target, dtype=np.float64)
    hand_target = np.asarray(hand_target, dtype=np.float64)

    for i in range(n_steps):
        t_norm = (i + 1) / n_steps
        s = 3 * t_norm**2 - 2 * t_norm**3   # cubic S-curve

        data.ctrl[0:N_ARM_JOINTS] = arm_start + s * (arm_target - arm_start)
        data.ctrl[N_ARM_JOINTS:N_ARM_JOINTS + N_HAND_JOINTS] = hand_target

        mujoco.mj_step(model, data)
        if log_fn is not None:
            log_fn(data)
        if viewer is not None:
            viewer.sync()


def _smooth_finger_close(model, data, arm_hold, hand_start, hand_target,
                         duration, log_fn=None, viewer=None):
    """Smoothly close fingers from hand_start to hand_target. Arm held still."""
    dt = model.opt.timestep
    n_steps = max(1, int(duration / dt))
    arm_hold = np.asarray(arm_hold, dtype=np.float64)
    hand_start = np.asarray(hand_start, dtype=np.float64)
    hand_target = np.asarray(hand_target, dtype=np.float64)

    for i in range(n_steps):
        t_norm = (i + 1) / n_steps
        s = 3 * t_norm**2 - 2 * t_norm**3

        data.ctrl[0:N_ARM_JOINTS] = arm_hold
        data.ctrl[N_ARM_JOINTS:N_ARM_JOINTS + N_HAND_JOINTS] = (
            hand_start + s * (hand_target - hand_start)
        )

        mujoco.mj_step(model, data)
        if log_fn is not None:
            log_fn(data)
        if viewer is not None:
            viewer.sync()


def _smooth_finger_close_thumb_first(model, data, arm_hold, hand_target,
                                     thumb_duration, fingers_duration,
                                     log_fn=None, viewer=None):
    """Sequential finger close: thumb first, then other fingers.

    Phase A (thumb_duration s):
        Thumb (joints 12, 13, 14) closes from 0 to hand_target.
        Other fingers (0-11) stay open at 0.

    Phase B (fingers_duration s):
        Thumb stays at target.
        Index, middle, ring, pinky (0-11) close from 0 to hand_target.

    This mimics the human grasp pattern: thumb forms the "wall" first,
    then fingers close against it. Reduces the likelihood of the hand
    pushing the object away on contact.
    """
    THUMB_IDX = slice(12, 15)   # thumb_cmc, thumb_mcp, thumb_ip
    FINGERS_IDX = slice(0, 12)  # index, middle, ring, pinky

    arm_hold = np.asarray(arm_hold, dtype=np.float64)
    hand_target = np.asarray(hand_target, dtype=np.float64)
    open_hand = np.zeros(N_HAND_JOINTS)

    dt = model.opt.timestep

    # ---- Phase A: thumb closes, fingers stay open ----
    n_steps_thumb = max(1, int(thumb_duration / dt))
    for i in range(n_steps_thumb):
        t_norm = (i + 1) / n_steps_thumb
        s = 3 * t_norm**2 - 2 * t_norm**3

        hand_state = open_hand.copy()
        hand_state[THUMB_IDX] = s * hand_target[THUMB_IDX]

        data.ctrl[0:N_ARM_JOINTS] = arm_hold
        data.ctrl[N_ARM_JOINTS:N_ARM_JOINTS + N_HAND_JOINTS] = hand_state

        mujoco.mj_step(model, data)
        if log_fn is not None:
            log_fn(data)
        if viewer is not None:
            viewer.sync()

    # ---- Phase B: fingers close, thumb held at target ----
    n_steps_fingers = max(1, int(fingers_duration / dt))
    for i in range(n_steps_fingers):
        t_norm = (i + 1) / n_steps_fingers
        s = 3 * t_norm**2 - 2 * t_norm**3

        hand_state = open_hand.copy()
        hand_state[THUMB_IDX] = hand_target[THUMB_IDX]            # hold thumb
        hand_state[FINGERS_IDX] = s * hand_target[FINGERS_IDX]    # close fingers

        data.ctrl[0:N_ARM_JOINTS] = arm_hold
        data.ctrl[N_ARM_JOINTS:N_ARM_JOINTS + N_HAND_JOINTS] = hand_state

        mujoco.mj_step(model, data)
        if log_fn is not None:
            log_fn(data)
        if viewer is not None:
            viewer.sync()


def _hold(model, data, duration, log_fn=None, viewer=None):
    """Hold current ctrl for `duration` seconds (no interpolation, just step)."""
    dt = model.opt.timestep
    n_steps = max(1, int(duration / dt))
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        if log_fn is not None:
            log_fn(data)
        if viewer is not None:
            viewer.sync()


def _compute_intermediate_pos(grasp_pos: np.ndarray, obj_pos: np.ndarray,
                              approach: str) -> np.ndarray:
    """Compute the intermediate waypoint position (offset from grasp_pos).

    For top approaches: offset is straight up in world +Z.
    For side approaches: offset is along the horizontal direction from
        the object to the grasp position (i.e., away from the object).
    """
    if approach == "top":
        return grasp_pos + np.array([0.0, 0.0, INTERMEDIATE_OFFSET])
    elif approach == "side":
        approach_dir = grasp_pos - obj_pos
        approach_dir[2] = 0.0   # horizontal component only
        norm = np.linalg.norm(approach_dir)
        if norm < 1e-6:
            # Object is directly under the grasp pose (degenerate) — fall back to top
            return grasp_pos + np.array([0.0, 0.0, INTERMEDIATE_OFFSET])
        approach_dir = approach_dir / norm
        return grasp_pos + approach_dir * INTERMEDIATE_OFFSET
    else:
        raise ValueError(f"Unknown approach type: {approach}")


# ============================================================================
# Main execution function
# ============================================================================

def execute_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    proposal: GraspProposal,
    record_log: bool = True,
    viewer=None,
) -> TrajectoryResult:
    """Execute a grasp proposal end-to-end with multi-waypoint trajectory.

    Args:
        model, data: MuJoCo model and data (data is mutated)
        proposal:    the grasp to attempt
        record_log:  if True, log per-timestep state for diagnostics
        viewer:      optional mujoco.viewer handle - if provided, sync() each step

    Returns:
        TrajectoryResult describing what happened
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proposal.object_name)
    if obj_bid < 0:
        raise ValueError(f"Object body '{proposal.object_name}' not found")

    initial_object_z = float(data.xpos[obj_bid, 2])
    obj_pos = data.xpos[obj_bid].copy()

    # Logging
    log = {"time": [], "qpos": [], "object_pos": []}
    def log_fn(d):
        if record_log:
            log["time"].append(float(d.time))
            log["qpos"].append(d.qpos.copy())
            log["object_pos"].append(d.xpos[obj_bid].copy())
    log_fn(data)

    # ------------------------------------------------------------------
    # Compute waypoint targets
    # ------------------------------------------------------------------
    grasp_pos = np.asarray(proposal.palm_pos, dtype=np.float64)
    grasp_quat = np.asarray(proposal.palm_quat, dtype=np.float64)
    grasp_R = _quat_to_matrix(grasp_quat)

    # Detect mug insert-and-grip approach
    is_mug_insert = (proposal.object_name == "mug" and proposal.approach == "top")

    # Pre-grasp: 15 cm above grasp, palm horizontal (facing down)
    pre_grasp_pos = grasp_pos.copy()
    pre_grasp_pos[2] += PRE_GRASP_HEIGHT
    pre_grasp_R = _quat_to_matrix(PALM_HORIZONTAL_QUAT)

    # Intermediate: a few cm before final grasp, in the grasp orientation
    intermediate_pos = _compute_intermediate_pos(grasp_pos, obj_pos, proposal.approach)
    intermediate_R = grasp_R

    # Lift: palm rises 15 cm in world Z, maintaining the grasp orientation.
    # With the thumb-up side orientation, the arm can maintain this pose
    # at higher Z without the kinematic issues of the old lateral orientation.
    lift_pos = grasp_pos.copy()
    lift_pos[2] += LIFT_DELTA
    lift_R = grasp_R

    finger_angles = np.asarray(proposal.finger_angles, dtype=np.float64)
    open_hand = np.zeros(N_HAND_JOINTS)
    q_home = data.qpos[:N_ARM_JOINTS].copy()

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------
    def make_result(success, failure_mode, **errs):
        if record_log and log["time"]:
            time_arr = np.array(log["time"])
            qpos_arr = np.array(log["qpos"])
            obj_arr = np.array(log["object_pos"])
        else:
            time_arr = np.zeros(0)
            qpos_arr = np.zeros((0, model.nq))
            obj_arr = np.zeros((0, 3))
        final_z = float(data.xpos[obj_bid, 2])
        return TrajectoryResult(
            object_name=proposal.object_name,
            grasp_type=proposal.grasp_type,
            approach=proposal.approach,
            success=success,
            failure_mode=failure_mode,
            pre_grasp_ik_pos_err=errs.get("pgp", -1.0),
            pre_grasp_ik_rot_err=errs.get("pgr", -1.0),
            intermediate_ik_pos_err=errs.get("ip", -1.0),
            intermediate_ik_rot_err=errs.get("ir", -1.0),
            grasp_ik_pos_err=errs.get("gp", -1.0),
            grasp_ik_rot_err=errs.get("gr", -1.0),
            lift_ik_pos_err=errs.get("lp", -1.0),
            lift_ik_rot_err=errs.get("lr", -1.0),
            initial_object_z=initial_object_z,
            final_object_z=final_z,
            lift_height=final_z - initial_object_z,
            time_log=time_arr,
            qpos_log=qpos_arr,
            object_pos_log=obj_arr,
        )

    # ------------------------------------------------------------------
    # IK solves (chained seeds for fast convergence)
    # ------------------------------------------------------------------
    q_pre, pgp, pgr = solve_ik_pose(
        model, data, pre_grasp_pos, pre_grasp_R,
        site_name=PALM_SITE, q_init=q_home, n_restarts=10,
        rot_tol=IK_ROT_TOL, pos_tol=IK_POS_TOL,
    )
    if q_pre is None:
        return make_result(False, "IK_FAIL_PREGRASP", pgp=pgp, pgr=pgr)

    q_int, ip, ir = solve_ik_pose(
        model, data, intermediate_pos, intermediate_R,
        site_name=PALM_SITE, q_init=q_pre, n_restarts=8,
        rot_tol=IK_ROT_TOL, pos_tol=IK_POS_TOL,
    )
    if q_int is None:
        return make_result(False, "IK_FAIL_INTERMEDIATE",
                           pgp=pgp, pgr=pgr, ip=ip, ir=ir)

    q_grasp, gp, gr = solve_ik_pose(
        model, data, grasp_pos, grasp_R,
        site_name=PALM_SITE, q_init=q_int, n_restarts=5,
        rot_tol=IK_ROT_TOL, pos_tol=IK_POS_TOL,
    )
    if q_grasp is None:
        return make_result(False, "IK_FAIL_GRASP",
                           pgp=pgp, pgr=pgr, ip=ip, ir=ir, gp=gp, gr=gr)

    # Mug: solve IK for push-down position (1cm lower, fingers seat inside mug)
    if is_mug_insert:
        push_down_pos = grasp_pos.copy()
        push_down_pos[2] -= 0.01
        q_push, _, _ = solve_ik_pose(
            model, data, push_down_pos, grasp_R,
            site_name=PALM_SITE, q_init=q_grasp, n_restarts=5,
            rot_tol=IK_ROT_TOL, pos_tol=IK_POS_TOL,
        )
        if q_push is None:
            q_push = q_grasp  # fallback: stay at grasp if push-down unreachable
    else:
        q_push = None  # not used for normal grasps

    q_lift, lp, lr = solve_ik_pose(
        model, data, lift_pos, lift_R,
        site_name=PALM_SITE, q_init=q_grasp, n_restarts=5,
        rot_tol=IK_ROT_TOL, pos_tol=IK_POS_TOL,
    )
    if q_lift is None:
        return make_result(False, "IK_FAIL_LIFT",
                           pgp=pgp, pgr=pgr, ip=ip, ir=ir,
                           gp=gp, gr=gr, lp=lp, lr=lr)

    # ------------------------------------------------------------------
    # Execute the trajectory
    # ------------------------------------------------------------------
    # Detect mug-specific insert approach (matches flag set during waypoint computation)
    # is_mug_insert was set above during waypoint computation

    # Compute hand states for mug insert
    if is_mug_insert:
        semi_closed = MUG_INSERT_FINGERS

    # PHASE 2: HOME -> PRE-GRASP (same for all objects)
    _smooth_motion(model, data,
                   arm_start=q_home, arm_target=q_pre,
                   hand_target=open_hand,
                   duration=DUR_HOME_TO_PREGRASP,
                   log_fn=log_fn, viewer=viewer)
    _hold(model, data, DUR_PHASE_SETTLE, log_fn=log_fn, viewer=viewer)

    # Knock-detection
    if data.xpos[obj_bid, 2] < initial_object_z - KNOCK_DETECTION_DROP:
        return make_result(False, "OBJECT_KNOCKED",
                           pgp=pgp, pgr=pgr, ip=ip, ir=ir,
                           gp=gp, gr=gr, lp=lp, lr=lr)

    if is_mug_insert:
        # ---- MUG INSERT-AND-GRIP SEQUENCE ----
        # Function A: semi-close fingers at pre-grasp
        _smooth_finger_close(model, data,
                             arm_hold=q_pre,
                             hand_start=open_hand,
                             hand_target=semi_closed,
                             duration=MUG_SEMI_CLOSE_DURATION,
                             log_fn=log_fn, viewer=viewer)

        # Descend to intermediate with semi-closed fingers
        _smooth_motion(model, data,
                       arm_start=q_pre, arm_target=q_int,
                       hand_target=semi_closed,
                       duration=DUR_PREGRASP_TO_INTERMEDIATE,
                       log_fn=log_fn, viewer=viewer)
        _hold(model, data, DUR_PHASE_SETTLE, log_fn=log_fn, viewer=viewer)

        # Insert into mug (intermediate -> grasp) with semi-closed fingers
        _smooth_motion(model, data,
                       arm_start=q_int, arm_target=q_grasp,
                       hand_target=semi_closed,
                       duration=DUR_INTERMEDIATE_TO_GRASP,
                       log_fn=log_fn, viewer=viewer)
        _hold(model, data, DUR_PHASE_SETTLE, log_fn=log_fn, viewer=viewer)

        # Push down 1cm to seat fingers inside the mug before gripping
        _smooth_motion(model, data,
                       arm_start=q_grasp, arm_target=q_push,
                       hand_target=semi_closed,
                       duration=0.5,
                       log_fn=log_fn, viewer=viewer)
        _hold(model, data, DUR_PHASE_SETTLE, log_fn=log_fn, viewer=viewer)

        # Function B: close middle + ring + thumb (grip inner wall)
        _smooth_finger_close(model, data,
                             arm_hold=q_push,
                             hand_start=semi_closed,
                             hand_target=MUG_GRIP_FINGERS,
                             duration=DUR_CLOSE_FINGERS,
                             log_fn=log_fn, viewer=viewer)

    else:
        # ---- NORMAL SEQUENCE (all other objects) ----
        # PHASE 3a: PRE-GRASP -> INTERMEDIATE
        _smooth_motion(model, data,
                       arm_start=q_pre, arm_target=q_int,
                       hand_target=open_hand,
                       duration=DUR_PREGRASP_TO_INTERMEDIATE,
                       log_fn=log_fn, viewer=viewer)
        _hold(model, data, DUR_PHASE_SETTLE, log_fn=log_fn, viewer=viewer)

        if data.xpos[obj_bid, 2] < initial_object_z - KNOCK_DETECTION_DROP:
            return make_result(False, "OBJECT_KNOCKED",
                               pgp=pgp, pgr=pgr, ip=ip, ir=ir,
                               gp=gp, gr=gr, lp=lp, lr=lr)

        # PHASE 3b: INTERMEDIATE -> GRASP
        _smooth_motion(model, data,
                       arm_start=q_int, arm_target=q_grasp,
                       hand_target=open_hand,
                       duration=DUR_INTERMEDIATE_TO_GRASP,
                       log_fn=log_fn, viewer=viewer)
        _hold(model, data, DUR_PHASE_SETTLE, log_fn=log_fn, viewer=viewer)

        # PHASE 4: CLOSE FINGERS
        _smooth_finger_close(model, data,
                             arm_hold=q_grasp,
                             hand_start=open_hand,
                             hand_target=finger_angles,
                             duration=DUR_CLOSE_FINGERS,
                             log_fn=log_fn, viewer=viewer)

    # PHASE 5: LIFT (fingers held in grasp configuration)
    lift_hand_target = MUG_GRIP_FINGERS if is_mug_insert else finger_angles
    lift_arm_start = q_push if is_mug_insert else q_grasp
    _smooth_motion(model, data,
                   arm_start=lift_arm_start, arm_target=q_lift,
                   hand_target=lift_hand_target,
                   duration=DUR_GRASP_TO_LIFT,
                   log_fn=log_fn, viewer=viewer)
    _hold(model, data, DUR_FINAL_SETTLE, log_fn=log_fn, viewer=viewer)

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------
    final_z = float(data.xpos[obj_bid, 2])
    lift_height = final_z - initial_object_z

    if lift_height >= LIFT_SUCCESS_HEIGHT:
        return make_result(True, None,
                           pgp=pgp, pgr=pgr, ip=ip, ir=ir,
                           gp=gp, gr=gr, lp=lp, lr=lr)
    else:
        return make_result(False, "GRASP_SLIP",
                           pgp=pgp, pgr=pgr, ip=ip, ir=ir,
                           gp=gp, gr=gr, lp=lp, lr=lr)


# ============================================================================
# Reachability-aware wrapper
# ============================================================================

def execute_best_proposal(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    proposals: list[GraspProposal],
    object_name: str,
    record_log: bool = True,
    viewer=None,
    verbose: bool = True,
) -> Optional[TrajectoryResult]:
    """Try proposals for `object_name` in score order, executing the first
    one whose pre-grasp IK is reachable.

    The reachability check uses the new horizontal pre-grasp pose (palm
    above grasp position, palm facing down) which is the same regardless
    of approach type. So it filters proposals whose grasp position is
    unreachable from above with a horizontal palm.
    """
    obj_props = [p for p in proposals if p.object_name == object_name]
    obj_props.sort(key=lambda p: p.score, reverse=True)

    if not obj_props:
        if verbose:
            print(f"  No proposals for object '{object_name}'")
        return None

    for proposal in obj_props:
        # Quick reachability check: horizontal pre-grasp above the grasp
        grasp_pos = np.asarray(proposal.palm_pos, dtype=np.float64)
        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += PRE_GRASP_HEIGHT
        pre_R = _quat_to_matrix(PALM_HORIZONTAL_QUAT)

        q_test, _, _ = solve_ik_pose(
            model, data, pre_grasp_pos, pre_R,
            site_name=PALM_SITE, n_restarts=10,
            rot_tol=IK_ROT_TOL, pos_tol=IK_POS_TOL,
        )
        if q_test is None:
            if verbose:
                print(f"  Skipping {proposal.grasp_type} {proposal.approach} "
                      f"(score {proposal.score:.2f}): pre-grasp IK unreachable")
            continue

        if verbose:
            print(f"  Executing {proposal.grasp_type} {proposal.approach} "
                  f"(score {proposal.score:.2f})")
        return execute_grasp(model, data, proposal,
                             record_log=record_log, viewer=viewer)

    if verbose:
        print(f"  No reachable proposal for '{object_name}'")
    return None