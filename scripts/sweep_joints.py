"""
Task 1 verification: load the combined Piper + RUKA model and sweep every joint
through its full range. Confirms the kinematic tree is wired correctly, joint
limits are sensible, and there is no gross interpenetration at the mount.

This v2 version uses a tightly coupled physics+viewer loop so the viewer
stays responsive during sweeps. The "current joint" being swept is updated
based on wall-clock time, not by blocking the main loop.

Usage:
    python scripts/sweep_joints.py
    python scripts/sweep_joints.py --robot          # robot only (no scene/floor)
    python scripts/sweep_joints.py --duration 3.0   # seconds per joint
    python scripts/sweep_joints.py --headless       # no viewer, just logs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = REPO_ROOT / "assets" / "mounted" / "piper_ruka_scene.xml"
ROBOT_XML = REPO_ROOT / "assets" / "mounted" / "piper_ruka.xml"


def effective_range(model, aid) -> tuple[float, float]:
    """Return the actuator's effective control range, falling back to the joint range."""
    if model.actuator_ctrllimited[aid]:
        return tuple(model.actuator_ctrlrange[aid])
    jid = model.actuator_trnid[aid, 0]
    if jid >= 0 and model.jnt_limited[jid]:
        return tuple(model.jnt_range[jid])
    return (-1.0, 1.0)


def print_model_summary(model: mujoco.MjModel) -> None:
    print("=" * 70)
    print(f"Model summary")
    print(f"  nq    : {model.nq}    (generalized coords)")
    print(f"  nv    : {model.nv}    (DoF)")
    print(f"  nu    : {model.nu}    (actuators)")
    print(f"  nbody : {model.nbody} (incl. world)")
    print(f"  njnt  : {model.njnt}")
    print("=" * 70)
    print("\nActuators with effective ranges used by the sweep:")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        lo, hi = effective_range(model, aid)
        src = "ctrl" if model.actuator_ctrllimited[aid] else "joint"
        print(f"  [{aid:2d}] {name:25s} [{lo:+.3f}, {hi:+.3f}]  (from {src} range)")
    print("=" * 70)


def home_pose(model):
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf_id < 0:
        raise RuntimeError("No 'home' keyframe in model.")
    return model.key_qpos[kf_id].copy(), model.key_ctrl[kf_id].copy()


def compute_target(model, current_aid: int, t_in_sweep: float, sweep_duration: float, home_ctrl):
    """Compute the full ctrl vector for the current sweep state."""
    ctrl = home_ctrl.copy()
    if current_aid is None or current_aid < 0 or current_aid >= model.nu:
        return ctrl
    lo, hi = effective_range(model, current_aid)
    phase = (t_in_sweep / sweep_duration) * 2 * np.pi
    s = np.sin(phase)
    target = (lo + hi) / 2 + s * (hi - lo) / 2
    ctrl[current_aid] = target
    return ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", action="store_true", help="Use robot-only XML (no floor)")
    ap.add_argument("--duration", type=float, default=2.5, help="Seconds per joint")
    ap.add_argument("--headless", action="store_true", help="No viewer")
    args = ap.parse_args()

    xml_path = ROBOT_XML if args.robot else SCENE_XML
    print(f"Loading {xml_path} ...")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    print_model_summary(model)

    qpos0, ctrl0 = home_pose(model)
    data.qpos[:] = qpos0
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)

    sweep_duration = args.duration
    n_actuators = model.nu
    total_sweep_time = sweep_duration * n_actuators
    print(f"\nSweeping {n_actuators} actuators x {sweep_duration:.1f}s = {total_sweep_time:.1f}s total\n")

    if args.headless:
        t_start = time.time()
        last_aid = -1
        while time.time() - t_start < total_sweep_time + 1.0:
            t = time.time() - t_start
            aid = int(t // sweep_duration)
            if aid >= n_actuators:
                aid = -1
            t_in = t - max(aid, 0) * sweep_duration
            if aid != last_aid and aid >= 0:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                print(f"  [{aid:2d}] sweeping {name}")
                last_aid = aid
            data.ctrl[:] = compute_target(model, aid, t_in, sweep_duration, ctrl0)
            mujoco.mj_step(model, data)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_start = time.time()
        last_aid = -1

        while viewer.is_running():
            t_real = time.time() - t_start

            if t_real < total_sweep_time:
                aid = int(t_real // sweep_duration)
                t_in = t_real - aid * sweep_duration
            else:
                aid = -1
                t_in = 0.0

            if aid != last_aid:
                if aid >= 0 and aid < n_actuators:
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                    lo, hi = effective_range(model, aid)
                    print(f"  [{aid:2d}] sweeping {name}  range=[{lo:+.3f}, {hi:+.3f}]")
                elif last_aid >= 0:
                    print("  done sweeping. Holding home pose. Close window to exit.")
                last_aid = aid

            data.ctrl[:] = compute_target(model, aid, t_in, sweep_duration, ctrl0)

            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()