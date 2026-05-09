"""
Task 2 deliverable: render RGB + depth + point cloud for every camera in
the scene, save everything to disk under outputs/render_<timestamp>/.

Usage:
    python scripts/render_scene.py
    python scripts/render_scene.py --width 1280 --height 720
    python scripts/render_scene.py --cameras wrist_cam        # subset
    python scripts/render_scene.py --no-fuse                  # skip fused cloud

Outputs (per run):
    outputs/render_<TS>/
        wrist_cam/
            rgb.png            # 640x480 RGB
            depth.png          # depth color-mapped for visualization
            depth.npy          # raw float depth in meters (H, W)
            cloud.ply          # world-frame colored point cloud
        scene_cam/  ...        # (same files)
        fused_cloud.ply        # combined point cloud from all cameras (if --fuse)
        cameras.json           # all camera intrinsics + extrinsics

The cameras.json file is the calibration export — Task 3's grasping pipeline
will read this to reproject world-frame grasp poses into image space, etc.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import matplotlib

# Use a non-interactive backend so we can render headless
matplotlib.use("Agg")

# Make src/ importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.perception.camera_render import (
    DEFAULT_HEIGHT, DEFAULT_WIDTH,
    list_cameras, load_scene, render_camera,
)
from src.perception.pointcloud import (
    fuse_pointclouds, save_pointcloud_ply, view_to_world_pointcloud,
)


SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"
OUTPUT_ROOT = REPO_ROOT / "outputs"


def colorize_depth(depth: np.ndarray, near: float = 0.1, far: float = 2.0) -> np.ndarray:
    """Map a (H, W) float depth image to an (H, W, 3) uint8 colorized image
    using the matplotlib 'turbo' colormap. Zero (invalid) pixels render black."""
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    # Normalize valid depths into [0, 1] using the given near/far
    d_norm = np.clip((depth - near) / (far - near + 1e-8), 0.0, 1.0)
    rgba = (matplotlib.colormaps["turbo"](d_norm) * 255).astype(np.uint8)
    rgba[~valid] = [0, 0, 0, 255]
    return rgba[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="Camera names to render (default: all)")
    ap.add_argument("--no-fuse", action="store_true",
                    help="Skip fused-pointcloud output")
    ap.add_argument("--settle", type=int, default=200,
                    help="Number of physics steps to settle objects before rendering")
    ap.add_argument("--depth-near", type=float, default=0.2,
                    help="Near depth (m) for depth visualization colormap")
    ap.add_argument("--depth-far", type=float, default=2.5,
                    help="Far depth (m) for depth visualization colormap")
    ap.add_argument("--out-name", type=str, default=None,
                    help="Output subdirectory name (default: timestamped)")
    args = ap.parse_args()

    # Output directory
    if args.out_name:
        out_dir = OUTPUT_ROOT / args.out_name
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = OUTPUT_ROOT / f"render_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene: {SCENE_XML}")
    model, data = load_scene(SCENE_XML, settle_steps=args.settle)

    available = list_cameras(model)
    if args.cameras:
        for c in args.cameras:
            if c not in available:
                raise SystemExit(
                    f"Camera '{c}' not found. Available: {available}"
                )
        cameras = list(args.cameras)
    else:
        cameras = available
    print(f"Rendering cameras: {cameras}")

    # Render each camera and save outputs
    per_view_clouds = []          # list of (points, colors) for fusion
    cameras_json = {"cameras": {}}
    for cam_name in cameras:
        print(f"  -> {cam_name}")
        view = render_camera(model, data, cam_name,
                             width=args.width, height=args.height)

        cam_dir = out_dir / cam_name
        cam_dir.mkdir(exist_ok=True)

        # RGB
        iio.imwrite(cam_dir / "rgb.png", view.rgb)
        # Depth: raw .npy + colorized .png
        np.save(cam_dir / "depth.npy", view.depth)
        depth_vis = colorize_depth(view.depth, args.depth_near, args.depth_far)
        iio.imwrite(cam_dir / "depth.png", depth_vis)

        # Point cloud
        pts, cols = view_to_world_pointcloud(view)
        save_pointcloud_ply(cam_dir / "cloud.ply", pts, cols)
        per_view_clouds.append((pts, cols))

        cameras_json["cameras"][cam_name] = view.info.to_dict()

        # Diagnostic per-camera summary
        n_valid = int((view.depth > 0).sum())
        n_total = view.depth.size
        depth_min = float(view.depth[view.depth > 0].min()) if n_valid else 0.0
        depth_max = float(view.depth.max())
        print(f"     valid pixels: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
        print(f"     depth range : [{depth_min:.3f}, {depth_max:.3f}] m")
        print(f"     point cloud : {pts.shape[0]} points → {cam_dir/'cloud.ply'}")

    # Fused cloud
    if not args.no_fuse and len(per_view_clouds) > 1:
        print(f"Fusing {len(per_view_clouds)} point clouds...")
        fpts, fcols = fuse_pointclouds(per_view_clouds)
        save_pointcloud_ply(out_dir / "fused_cloud.ply", fpts, fcols)
        cameras_json["fused"] = {"num_points": int(fpts.shape[0])}
        print(f"  fused cloud: {fpts.shape[0]} points → {out_dir/'fused_cloud.ply'}")

    # Camera calibration JSON
    with open(out_dir / "cameras.json", "w") as f:
        json.dump(cameras_json, f, indent=2)
    print(f"Wrote calibration: {out_dir/'cameras.json'}")

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()