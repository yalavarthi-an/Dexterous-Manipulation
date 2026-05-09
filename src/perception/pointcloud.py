"""
Point cloud generation and fusion from rendered depth images.

Conventions (matches camera_render.get_camera_info):
  - Camera frame: OpenCV style. +Z is the look direction (into the scene),
    +X is image-right, +Y is image-down.
  - Depth values: distance in meters along the camera's +Z axis.
  - Zero depth means "no measurement" (sky or beyond the far plane).
  - World frame: MuJoCo's world (Z up).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera_render import CameraInfo, RenderedView


# ============================================================================
# Depth -> point cloud
# ============================================================================

def depth_to_points_camera_frame(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Deproject a depth image to 3D points in the camera frame.

    Args:
        depth: (H, W) float array, distance in meters along camera +Z.
               Zeros are treated as "no measurement" and dropped.
        K: 3x3 intrinsic matrix.

    Returns:
        (N, 3) array of points in camera frame, where N <= H*W (zeros dropped).

    Math:
        For a pixel (u, v) with depth Z:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = Z
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Build a meshgrid of pixel coordinates (u along X, v along Y)
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)  # both (H, W)

    # Vectorized deprojection
    z = depth                                        # (H, W)
    x = (uu - cx) * z / fx                           # (H, W)
    y = (vv - cy) * z / fy                           # (H, W)
    points = np.stack([x, y, z], axis=-1)            # (H, W, 3)

    # Drop zero-depth pixels
    valid = z > 0.0
    return points[valid]


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N, 3) points → (N, 3) points."""
    if points.size == 0:
        return points
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # (N, 4)
    out = (T @ points_h.T).T                                                    # (N, 4)
    return out[:, :3]


def view_to_world_pointcloud(view: RenderedView) -> tuple[np.ndarray, np.ndarray]:
    """Produce a (points, colors) world-frame point cloud from a RenderedView.

    Returns:
        points: (N, 3) float64, in world coords
        colors: (N, 3) uint8, RGB matching points (kept aligned via the same valid mask)
    """
    h, w = view.depth.shape
    fx, fy = view.info.K[0, 0], view.info.K[1, 1]
    cx, cy = view.info.K[0, 2], view.info.K[1, 2]

    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    z = view.depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    valid = z > 0.0
    pts_cam = np.stack([x, y, z], axis=-1)[valid]  # (N, 3)
    cols    = view.rgb[valid]                       # (N, 3) uint8

    pts_world = transform_points(pts_cam, view.info.T_world_cam)
    return pts_world, cols


def fuse_pointclouds(per_view: list[tuple[np.ndarray, np.ndarray]]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate multiple (points, colors) point clouds into one."""
    if not per_view:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    all_pts  = np.concatenate([p for p, _ in per_view], axis=0)
    all_cols = np.concatenate([c for _, c in per_view], axis=0)
    return all_pts, all_cols


# ============================================================================
# I/O: write .ply files
# ============================================================================

def save_pointcloud_ply(path: Path | str, points: np.ndarray,
                        colors: np.ndarray | None = None) -> None:
    """Write a colored point cloud to ASCII .ply format.

    No external dependencies — Open3D and trimesh both read this format,
    as do MeshLab, CloudCompare, and the standard ROS rviz pcd viewer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = points.shape[0]
    has_color = colors is not None and colors.shape[0] == n

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if has_color:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
