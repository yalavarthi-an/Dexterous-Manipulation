"""
Task 3 deliverable: generate and visualize dexterous grasp proposals.

Loads the full scene, renders point clouds, segments objects, proposes
grasps for each, and opens an Open3D viewer showing the top-K grasps
overlaid on the scene point cloud.

Each grasp is visualized as:
  - A coordinate frame at the palm target position (RGB = XYZ axes)
  - An arrow from pre-grasp to grasp position (the approach trajectory)
  - A text label (printed to console) with object name + grasp type

Usage:
    python scripts/visualize_grasps.py
    python scripts/visualize_grasps.py --top-k 5     # show top 5 grasps
    python scripts/visualize_grasps.py --save         # save proposals to JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.perception.camera_render import load_scene, render_all_cameras
from src.perception.pointcloud import fuse_pointclouds, view_to_world_pointcloud
from src.grasping.grasp_proposal import (
    get_object_positions, segment_objects, propose_grasps, GraspProposal,
)

SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert (w,x,y,z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def make_grasp_frame(proposal: GraspProposal, size: float = 0.06) -> o3d.geometry.TriangleMesh:
    """Create a coordinate-frame mesh at the grasp palm pose."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    R = quat_to_matrix(proposal.palm_quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = proposal.palm_pos
    frame.transform(T)
    return frame


def make_approach_arrow(proposal: GraspProposal, color=(1, 0.5, 0)) -> list:
    """Create a line showing the approach trajectory (pre-grasp → grasp)."""
    pts = [proposal.pre_grasp_pos.tolist(), proposal.palm_pos.tolist()]
    lines = [[0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])

    # Add a small sphere at the grasp point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(proposal.palm_pos)
    sphere.paint_uniform_color(color)

    return [line_set, sphere]


def make_obb_wireframe(obj_info, color=(0, 1, 0)) -> o3d.geometry.LineSet:
    """Create a wireframe box showing the object's OBB."""
    # 8 corners of the OBB
    ext = obj_info.obb_extents
    R = obj_info.obb_R
    c = obj_info.obb_center
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corner = c + R @ np.array([sx * ext[0], sy * ext[1], sz * ext[2]])
                corners.append(corner)

    # 12 edges of a box
    edges = [
        [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],
        [2,6],[3,7],[4,5],[4,6],[5,7],[6,7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10, help="Show top K grasps")
    ap.add_argument("--save", action="store_true", help="Save proposals to JSON")
    ap.add_argument("--camera", default="scene_cam", help="Camera for rendering")
    args = ap.parse_args()

    print("Loading scene...")
    model, data = load_scene(SCENE_XML, settle_steps=300)

    print("Rendering cameras...")
    views = render_all_cameras(model, data)

    print("Building point clouds...")
    per_view = [view_to_world_pointcloud(v) for v in views]
    all_pts, all_cols = fuse_pointclouds(per_view)
    print(f"  fused cloud: {all_pts.shape[0]} points")

    print("Segmenting objects...")
    obj_positions = get_object_positions(model, data)
    objects = segment_objects(all_pts, all_cols, obj_positions)
    print(f"  segmented {len(objects)} objects:")
    for obj in objects:
        print(f"    {obj.name:18s}  {obj.points.shape[0]:5d} points  "
              f"extents=[{obj.obb_extents[0]:.3f}, {obj.obb_extents[1]:.3f}, {obj.obb_extents[2]:.3f}]")

    print("Proposing grasps...")
    proposals = propose_grasps(objects)
    top_proposals = proposals[:args.top_k]
    print(f"  generated {len(proposals)} proposals, showing top {len(top_proposals)}:")
    for i, p in enumerate(top_proposals):
        print(f"    [{i+1}] {p.object_name:18s}  {p.grasp_type:18s}  {p.approach:5s}  "
              f"score={p.score:.3f}  palm={p.palm_pos}")

    if args.save:
        out_path = REPO_ROOT / "outputs" / "grasp_proposals.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([p.to_dict() for p in proposals], f, indent=2)
        print(f"  saved to {out_path}")

    # Build visualization
    print("\nBuilding visualization...")
    geometries = []

    # Scene point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols.astype(np.float64) / 255.0)
    geometries.append(pcd)

    # World axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    geometries.append(axes)

    # Per-object OBB wireframes
    for obj in objects:
        obb_wf = make_obb_wireframe(obj, color=(0, 0.8, 0))
        geometries.append(obb_wf)

    # Grasp proposals
    COLORS = [
        (1, 0, 0), (0, 0, 1), (1, 0.5, 0), (0.5, 0, 1), (0, 0.8, 0.8),
        (1, 1, 0), (1, 0, 1), (0.5, 0.5, 0), (0, 0.5, 1), (1, 0.5, 0.5),
    ]
    for i, p in enumerate(top_proposals):
        color = COLORS[i % len(COLORS)]
        frame = make_grasp_frame(p, size=0.05)
        geometries.append(frame)
        arrow_geoms = make_approach_arrow(p, color=color)
        geometries.extend(arrow_geoms)

    print(f"\nOpening viewer with {len(geometries)} geometries.")
    print("  Green wireframes = object OBBs")
    print("  Colored frames   = grasp palm poses (RGB = XYZ axes)")
    print("  Orange arrows    = approach trajectories (pre-grasp → grasp)")
    print("  Close window to exit.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Grasp Proposals",
        width=1400, height=900,
    )


if __name__ == "__main__":
    main()
