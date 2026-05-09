"""
Quick viewer for a saved .ply point cloud.

Usage:
    python scripts/view_pointcloud.py outputs/render_2026-05-04_19-52-12/fused_cloud.ply
    python scripts/view_pointcloud.py outputs/render_LATEST/fused_cloud.ply
    python scripts/view_pointcloud.py                    # picks latest fused_cloud.ply

Renders an interactive Open3D viewer. Use mouse to rotate, scroll to zoom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"


def find_latest_fused() -> Path:
    """Find the most recent outputs/render_*/fused_cloud.ply."""
    candidates = sorted(OUTPUTS_DIR.glob("render_*/fused_cloud.ply"))
    if not candidates:
        raise FileNotFoundError("No fused_cloud.ply found in outputs/render_*/")
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None,
                    help=".ply path. If omitted, uses latest fused cloud.")
    ap.add_argument("--with-axes", action="store_true",
                    help="Also draw the world-frame coordinate axes.")
    args = ap.parse_args()

    if args.path:
        ply_path = Path(args.path)
    else:
        ply_path = find_latest_fused()
    print(f"Loading {ply_path} ...")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"Loaded {len(pcd.points)} points")

    if len(pcd.points) == 0:
        print("Empty point cloud!")
        return

    # Bounding box info — sanity check that points are in world frame
    pts = np.asarray(pcd.points)
    print(f"Bounding box (world frame, m):")
    print(f"  x: [{pts[:,0].min():+.3f}, {pts[:,0].max():+.3f}]")
    print(f"  y: [{pts[:,1].min():+.3f}, {pts[:,1].max():+.3f}]")
    print(f"  z: [{pts[:,2].min():+.3f}, {pts[:,2].max():+.3f}]")

    geometries = [pcd]
    if args.with_axes:
        # Add world-frame axes at origin (red=X, green=Y, blue=Z), 0.3m long
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        geometries.append(axes)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"PointCloud: {ply_path.name}",
        width=1200, height=800,
    )


if __name__ == "__main__":
    main()
