"""
Run a grasp execution on a single object.

Usage:
    python scripts/run_grasp.py --object banana
    python scripts/run_grasp.py --object mug --headless
    python scripts/run_grasp.py --object tennis_ball --proposals outputs/grasp_proposals.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.perception.camera_render import load_scene, render_all_cameras
from src.perception.pointcloud import view_to_world_pointcloud, fuse_pointclouds
from src.grasping.grasp_proposal import (
    get_object_positions, segment_objects, propose_grasps, GraspProposal
)
from src.planning.executor import execute_best_proposal

SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"
DEFAULT_PROPOSALS = REPO_ROOT / "outputs" / "grasp_proposals.json"


def proposals_from_json(path: Path) -> list[GraspProposal]:
    """Reconstruct GraspProposal objects from saved JSON."""
    with open(path) as f:
        data = json.load(f)
    return [
        GraspProposal(
            object_name=p["object_name"],
            grasp_type=p["grasp_type"],
            approach=p["approach"],
            description=p.get("description", ""),
            palm_pos=np.array(p["palm_pos"]),
            palm_quat=np.array(p["palm_quat"]),
            pre_grasp_pos=np.array(p["pre_grasp_pos"]),
            finger_angles=np.array(p["finger_angles"]),
            score=p["score"],
        )
        for p in data
    ]


def regenerate_proposals(model, data) -> list[GraspProposal]:
    """Run perception + grasp proposal pipeline from scratch."""
    print("  Rendering cameras...")
    views = render_all_cameras(model, data)
    print("  Building point clouds...")
    per_view = [view_to_world_pointcloud(v) for v in views]
    all_pts, all_cols = fuse_pointclouds(per_view)
    print(f"  Fused cloud: {all_pts.shape[0]} points")
    print("  Segmenting objects...")
    obj_positions = get_object_positions(model, data)
    objects = segment_objects(all_pts, all_cols, obj_positions)
    print(f"  Segmented {len(objects)} objects")
    return propose_grasps(objects)


def print_result(result):
    print("\n" + "=" * 60)
    print(f"Result: {result.object_name}")
    print("=" * 60)
    print(f"  Grasp type:      {result.grasp_type} ({result.approach})")
    icon = "OK" if result.success else "FAIL"
    print(f"  Success:         {icon}  {result.success}")
    if result.failure_mode:
        print(f"  Failure mode:    {result.failure_mode}")
    print(f"  Initial obj Z:   {result.initial_object_z:.3f} m")
    print(f"  Final obj Z:     {result.final_object_z:.3f} m")
    print(f"  Lift height:     {result.lift_height*1000:+.1f} mm")
    print()
    print(f"  IK errors (mm / deg):")
    if result.pre_grasp_ik_pos_err >= 0:
        print(f"    Pre-grasp:     {result.pre_grasp_ik_pos_err*1000:6.1f} mm  /  "
              f"{np.degrees(result.pre_grasp_ik_rot_err):5.1f} deg")
    if result.intermediate_ik_pos_err >= 0:
        print(f"    Intermediate:  {result.intermediate_ik_pos_err*1000:6.1f} mm  /  "
              f"{np.degrees(result.intermediate_ik_rot_err):5.1f} deg")
    if result.grasp_ik_pos_err >= 0:
        print(f"    Grasp:         {result.grasp_ik_pos_err*1000:6.1f} mm  /  "
              f"{np.degrees(result.grasp_ik_rot_err):5.1f} deg")
    if result.lift_ik_pos_err >= 0:
        print(f"    Lift:          {result.lift_ik_pos_err*1000:6.1f} mm  /  "
              f"{np.degrees(result.lift_ik_rot_err):5.1f} deg")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True, help="YCB object to grasp")
    ap.add_argument("--headless", action="store_true",
                    help="Run without viewer (faster)")
    ap.add_argument("--proposals", default=None,
                    help="Path to grasp proposals JSON (default: regenerate)")
    args = ap.parse_args()

    print("Loading scene...")
    model, data = load_scene(SCENE_XML, settle_steps=300)

    # Get proposals
    if args.proposals:
        path = Path(args.proposals)
        print(f"Loading proposals from {path}")
        proposals = proposals_from_json(path)
    elif DEFAULT_PROPOSALS.exists():
        print(f"Loading proposals from {DEFAULT_PROPOSALS}")
        proposals = proposals_from_json(DEFAULT_PROPOSALS)
    else:
        print("Generating fresh proposals...")
        proposals = regenerate_proposals(model, data)

    print(f"\nLoaded {len(proposals)} proposals total")
    obj_props = [p for p in proposals if p.object_name == args.object]
    print(f"Found {len(obj_props)} proposals for '{args.object}'")
    if not obj_props:
        avail = sorted({p.object_name for p in proposals})
        print(f"Available objects: {avail}")
        sys.exit(1)

    print(f"\nExecuting grasp on {args.object}...")
    t0 = time.time()

    if args.headless:
        result = execute_best_proposal(model, data, proposals, args.object)
    else:
        try:
            import mujoco.viewer
        except ImportError:
            print("mujoco.viewer not available - running headless")
            result = execute_best_proposal(model, data, proposals, args.object)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                result = execute_best_proposal(
                    model, data, proposals, args.object, viewer=viewer,
                )
                if result is not None:
                    print("\n  Execution complete. Close the viewer to exit.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.01)

    elapsed = time.time() - t0
    print(f"\n[Done in {elapsed:.1f}s wall time]")

    if result is None:
        print(f"FAIL: No reachable proposal for {args.object}")
        sys.exit(1)

    print_result(result)


if __name__ == "__main__":
    main()