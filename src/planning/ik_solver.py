"""
Inverse kinematics solver for the Piper arm.

Uses damped least-squares (Levenberg-Marquardt) IK with MuJoCo's
analytical Jacobian. Solves for the 6 arm joint angles only — finger
joints are not modified.

Two modes:
  - Position-only (3-DOF target): solve_ik_position()
  - Full pose (6-DOF target):     solve_ik_pose()

The solver operates on a COPY of the simulation data so it does not
disturb the running sim state. Call the solver, get joint angles back,
then apply them to the real data.ctrl in the execution loop.
"""

from __future__ import annotations

import mujoco
import numpy as np


# Number of arm joints (first 6 actuators in our model)
N_ARM_JOINTS = 6


def _get_site_pose(model: mujoco.MjModel, data: mujoco.MjData,
                   site_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, rotation_matrix) of a named site in world frame."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found")
    pos = data.site_xpos[sid].copy()
    rot = data.site_xmat[sid].reshape(3, 3).copy()
    return pos, rot


def _get_site_jacobian(model: mujoco.MjModel, data: mujoco.MjData,
                       site_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (Jp, Jr) — positional and rotational Jacobians of a site.

    Each is sliced to only the first N_ARM_JOINTS columns (the Piper arm),
    ignoring the 15 finger joints.

    Returns:
        Jp: (3, 6) positional Jacobian
        Jr: (3, 6) rotational Jacobian
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    return jacp[:, :N_ARM_JOINTS], jacr[:, :N_ARM_JOINTS]


def _orientation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Compute orientation error as a 3D rotation vector (axis * angle)."""
    R_err = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2 * np.sin(angle) + 1e-10)
    return axis * angle


def _clamp_to_limits(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    """Clamp the first N_ARM_JOINTS joint values to their limits."""
    q = q.copy()
    for i in range(N_ARM_JOINTS):
        if model.jnt_limited[i]:
            lo, hi = model.jnt_range[i]
            q[i] = np.clip(q[i], lo, hi)
    return q


# ============================================================================
# Position-only IK (3-DOF target, 6 arm joints)
# ============================================================================

def solve_ik_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    site_name: str = "flange",
    max_iters: int = 200,
    tol: float = 0.005,
    damping: float = 0.05,
    step_size: float = 0.3,
    q_init: np.ndarray | None = None,
    n_restarts: int = 5,
) -> tuple[np.ndarray | None, float]:
    """Solve position-only IK for the arm with multi-start.

    Tries from the given q_init first, then from random initial configs.
    Returns the best solution found across all starts.

    Returns:
        (q_solution, final_error) if converged, or (None, final_error) if failed
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    best_q = None
    best_err = float("inf")

    # Build list of initial configs to try
    inits = []
    if q_init is not None:
        inits.append(q_init.copy())
    else:
        inits.append(data.qpos[:N_ARM_JOINTS].copy())

    # Add random restarts (sample within joint limits)
    rng = np.random.default_rng(42)
    for _ in range(n_restarts - 1):
        q_rand = np.zeros(N_ARM_JOINTS)
        for j in range(N_ARM_JOINTS):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                q_rand[j] = rng.uniform(lo, hi)
        inits.append(q_rand)

    for q0 in inits:
        d = mujoco.MjData(model)
        d.qpos[:] = data.qpos[:]
        d.qvel[:] = 0
        d.ctrl[:] = data.ctrl[:]
        d.qpos[:N_ARM_JOINTS] = q0

        for i in range(max_iters):
            mujoco.mj_forward(model, d)
            current_pos, _ = _get_site_pose(model, d, site_name)
            error = target_pos - current_pos
            err_norm = np.linalg.norm(error)

            if err_norm < tol:
                return d.qpos[:N_ARM_JOINTS].copy(), err_norm

            if err_norm < best_err:
                best_err = err_norm
                best_q = d.qpos[:N_ARM_JOINTS].copy()

            Jp, _ = _get_site_jacobian(model, d, site_name)
            JJT = Jp @ Jp.T + damping**2 * np.eye(3)
            dq = Jp.T @ np.linalg.solve(JJT, error)

            d.qpos[:N_ARM_JOINTS] += step_size * dq
            d.qpos[:N_ARM_JOINTS] = _clamp_to_limits(model, d.qpos[:N_ARM_JOINTS])

    # Return best found (even if not converged)
    if best_q is not None and best_err < tol * 5:
        return best_q, best_err
    return None, best_err


# ============================================================================
# Full 6-DOF IK (position + orientation)
# ============================================================================

def solve_ik_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    site_name: str = "flange",
    max_iters: int = 800,
    pos_tol: float = 0.005,
    rot_tol: float = 0.08,
    damping: float = 0.05,
    step_size: float = 0.2,
    pos_weight: float = 1.0,
    rot_weight: float = 0.5,
    q_init: np.ndarray | None = None,
    n_restarts: int = 15,
) -> tuple[np.ndarray | None, float, float]:
    """Solve full 6-DOF IK (position + orientation) with multi-start.

    Returns:
        (q_solution, pos_error, rot_error) or (None, pos_error, rot_error)
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_rot = np.asarray(target_rot, dtype=np.float64)

    best_q = None
    best_cost = float("inf")
    best_pos_err = float("inf")
    best_rot_err = float("inf")

    inits = []
    if q_init is not None:
        inits.append(q_init.copy())
    else:
        inits.append(data.qpos[:N_ARM_JOINTS].copy())

    rng = np.random.default_rng(42)
    for _ in range(n_restarts - 1):
        q_rand = np.zeros(N_ARM_JOINTS)
        for j in range(N_ARM_JOINTS):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                q_rand[j] = rng.uniform(lo, hi)
        inits.append(q_rand)

    for q0 in inits:
        d = mujoco.MjData(model)
        d.qpos[:] = data.qpos[:]
        d.qvel[:] = 0
        d.ctrl[:] = data.ctrl[:]
        d.qpos[:N_ARM_JOINTS] = q0

        for i in range(max_iters):
            mujoco.mj_forward(model, d)
            cur_pos, cur_rot = _get_site_pose(model, d, site_name)

            pos_err = target_pos - cur_pos
            rot_err = _orientation_error(cur_rot, target_rot)

            pos_norm = np.linalg.norm(pos_err)
            rot_norm = np.linalg.norm(rot_err)
            cost = pos_weight * pos_norm + rot_weight * rot_norm

            if cost < best_cost:
                best_cost = cost
                best_q = d.qpos[:N_ARM_JOINTS].copy()
                best_pos_err = pos_norm
                best_rot_err = rot_norm

            if pos_norm < pos_tol and rot_norm < rot_tol:
                return d.qpos[:N_ARM_JOINTS].copy(), pos_norm, rot_norm

            error = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])
            Jp, Jr = _get_site_jacobian(model, d, site_name)
            J = np.vstack([pos_weight * Jp, rot_weight * Jr])

            JJT = J @ J.T + damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, error)

            d.qpos[:N_ARM_JOINTS] += step_size * dq
            d.qpos[:N_ARM_JOINTS] = _clamp_to_limits(model, d.qpos[:N_ARM_JOINTS])

    if best_q is not None and best_pos_err < pos_tol * 8 and best_rot_err < rot_tol * 8:
        return best_q, best_pos_err, best_rot_err
    return None, best_pos_err, best_rot_err