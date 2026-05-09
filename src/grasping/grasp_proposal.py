"""
Heuristic dexterous grasp proposal for the Piper+RUKA system.

Pipeline:
  1. Query MuJoCo for each object's world-frame position and bounding radius
  2. Crop the scene point cloud to each object's neighborhood
  3. Compute the object's oriented bounding box (OBB) from its point cluster
  4. Look up the object's grasp config (type + approach direction)
  5. Compute the palm pose in world frame based on approach + OBB
  6. Return a ranked list of GraspProposal objects

This is a heuristic pipeline: it uses geometric reasoning (OBB principal axes,
approach-direction templates) rather than a learned model. The brief explicitly
endorses this approach when well-justified.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import open3d as o3d

from .presets import load_presets, load_object_grasp_config


# ============================================================================
# Data structures
# ============================================================================

@dataclasses.dataclass
class ObjectInfo:
    """Segmented object with its point cluster and bounding box."""
    name: str
    center: np.ndarray        # (3,) world-frame centroid
    points: np.ndarray        # (N, 3) world-frame points
    colors: np.ndarray        # (N, 3) uint8 RGB
    obb_center: np.ndarray    # (3,) OBB center
    obb_extents: np.ndarray   # (3,) half-extents sorted [longest, mid, shortest]
    obb_axes: np.ndarray      # (3, 3) rows are principal axes (longest first)
    obb_R: np.ndarray         # (3, 3) rotation matrix of the OBB


@dataclasses.dataclass
class GraspProposal:
    """A proposed grasp: where to put the palm, how to close the fingers."""
    object_name: str
    grasp_type: str           # preset name from grasp_presets.yaml
    approach: str             # "top" or "side"
    description: str          # human-readable
    palm_pos: np.ndarray      # (3,) world-frame palm target position
    palm_quat: np.ndarray     # (4,) world-frame palm target orientation (w,x,y,z)
    pre_grasp_pos: np.ndarray # (3,) pre-grasp position (above/behind the grasp point)
    finger_angles: np.ndarray # (15,) RUKA joint angles for this grasp type
    score: float              # heuristic score (higher = better)

    def to_dict(self) -> dict:
        return {
            "object_name": self.object_name,
            "grasp_type": self.grasp_type,
            "approach": self.approach,
            "description": self.description,
            "palm_pos": self.palm_pos.tolist(),
            "palm_quat": self.palm_quat.tolist(),
            "pre_grasp_pos": self.pre_grasp_pos.tolist(),
            "finger_angles": self.finger_angles.tolist(),
            "score": float(self.score),
        }


# ============================================================================
# Object segmentation (using known positions from MuJoCo)
# ============================================================================

OBJECT_NAMES = ["banana", "mug", "cracker_box", "mustard_bottle", "tennis_ball"]

# Approximate radii for point-cloud cropping (meters)
CROP_RADIUS = {
    "banana":         0.12,
    "mug":            0.10,
    "cracker_box":    0.18,
    "mustard_bottle": 0.14,
    "tennis_ball":    0.06,
}

# Table height for filtering
TABLE_Z = 0.70

# Per-object clearance overrides (meters). Objects not listed use the
# function default (0.0225 = 2.25 cm).
CLEARANCE_OVERRIDES = {
    "banana": 0.03,    # 3.0 cm
    "mug":   -0.01,    # knuckle line 1cm BELOW mug top (inside the opening)
}


def get_object_positions(model: mujoco.MjModel, data: mujoco.MjData
                         ) -> dict[str, np.ndarray]:
    """Read the world-frame position of each YCB object body."""
    positions = {}
    for name in OBJECT_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            positions[name] = data.xpos[bid].copy()
    return positions


def segment_objects(points: np.ndarray, colors: np.ndarray,
                    object_positions: dict[str, np.ndarray]
                    ) -> list[ObjectInfo]:
    """Segment the scene point cloud into per-object clusters.

    Strategy: for each known object position, crop points within CROP_RADIUS
    and above the table surface. Then compute the OBB of the cluster.
    """
    results = []
    for name, obj_pos in object_positions.items():
        radius = CROP_RADIUS.get(name, 0.12)

        # Crop: points within radius of object center AND above the table
        dists = np.linalg.norm(points - obj_pos, axis=1)
        mask = (dists < radius) & (points[:, 2] > TABLE_Z + 0.01)

        obj_pts = points[mask]
        obj_cols = colors[mask]

        if obj_pts.shape[0] < 20:
            # Too few points — object not visible from this viewpoint
            continue

        # Compute OBB via PCA
        centroid = obj_pts.mean(axis=0)
        centered = obj_pts - centroid
        cov = centered.T @ centered / centered.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by descending eigenvalue (longest axis first)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Half-extents from eigenvalues (standard deviation scaled to cover ~95%)
        extents = 2.0 * np.sqrt(eigenvalues + 1e-8)  # approximate half-extent

        # More accurate: project points onto axes and compute actual extent
        projected = centered @ eigenvectors  # (N, 3)
        extents = (projected.max(axis=0) - projected.min(axis=0)) / 2.0
        obb_center = centroid + eigenvectors @ (
            (projected.max(axis=0) + projected.min(axis=0)) / 2.0
        )

        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        results.append(ObjectInfo(
            name=name,
            center=obj_pos,
            points=obj_pts,
            colors=obj_cols,
            obb_center=obb_center,
            obb_extents=extents,
            obb_axes=eigenvectors.T,  # rows = axes
            obb_R=eigenvectors,       # columns = axes
        ))

    return results


# ============================================================================
# Grasp proposal
# ============================================================================

def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion."""
    # Shepperd's method
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w,x,y,z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# The mount quaternion from piper_ruka.xml: ruka_palm relative to link6
MOUNT_QUAT = np.array([0.0, 0.707, 0.707, 0.0])


def _link6_quat_to_palm_quat(link6_quat: np.ndarray) -> np.ndarray:
    """Convert a desired link6 world-frame quaternion to the corresponding
    ruka_palm world-frame quaternion by composing with the mount transform.

    The IK solver targets the palm body, not link6. So if we know what
    orientation link6 should have, we compose:
        palm_quat_world = link6_quat_world * mount_quat
    to get the palm's world orientation that the IK should target.
    """
    q = _quat_multiply(link6_quat, MOUNT_QUAT)
    if q[0] < 0:
        q = -q  # ensure w > 0 (canonical form)
    return q / np.linalg.norm(q)


def _compute_palm_pose_top(obj_info: ObjectInfo, clearance: float = 0.0225
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute palm pose for a top-down approach.

    Uses the VERTICAL extent of the OBB (the axis most aligned with world Z)
    to compute the object's actual top, not the largest extent (which could
    be the object's length for objects lying on their side, like a banana).
    """
    # Find which OBB axis is most vertical (closest to world Z)
    z_components = np.abs(obj_info.obb_axes[:, 2])  # Z component of each axis
    vertical_idx = np.argmax(z_components)
    z_extent = obj_info.obb_extents[vertical_idx]

    obj_top_z = obj_info.obb_center[2] + z_extent
    palm_z = obj_top_z + clearance

    palm_pos = np.array([
        obj_info.obb_center[0],
        obj_info.obb_center[1],
        palm_z,
    ])

    # Pre-grasp: 8cm above the grasp point
    pre_grasp_pos = palm_pos.copy()
    pre_grasp_pos[2] += 0.08

    # Desired link6 orientation for top-down (natural "reach over" pose):
    #   link6 +X (palm-out)  → world -Z   (palm faces DOWN toward the object)
    #   link6 +Y (thumb side) → world +Y  (perpendicular to approach)
    #   link6 +Z (fingers)   → world +X   (fingers extend FORWARD over the object)
    # This corresponds to a rotation matrix:
    #     R = [[ 0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    # which is a 90° rotation around world +Y axis. Quaternion:
    #     (cos(45°), 0, sin(45°), 0) = (0.707, 0, 0.707, 0)
    # The IK target site "palm" is in link6's frame, so this quaternion
    # IS the palm_quat — no mount-transform composition needed.
    palm_quat = np.array([0.7071068, 0.0, 0.7071068, 0.0])

    return palm_pos, palm_quat, pre_grasp_pos


def _compute_palm_pose_side(obj_info: ObjectInfo, clearance: float = 0.012, thumb_down: bool = False
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute palm pose for a lateral/side approach along Y axis.

    Smart approach direction: the hand always approaches from the Y side
    closest to Y=0 (robot center). This puts the palm between the object
    and the robot, which is always more reachable than going to the far side.

    Objects at +Y: hand approaches from -Y side (moves toward +Y)
    Objects at -Y: hand approaches from +Y side (moves toward -Y)
    """
    obj_center = obj_info.obb_center.copy()

    # Pick approach direction that keeps palm between object and robot center
    if obj_center[1] > 0:
        # Object on +Y side → palm on -Y side → approach toward +Y
        approach_offset = np.array([0.0, -1.0, 0.0])
        # Palm faces +Y (toward object), THUMB UP (+Z), fingers forward (+X)
        # link6+X=world+Y, link6+Y=world+Z, link6+Z=world+X
        # This is a 120° rotation around (1,1,1)/sqrt(3)
        palm_quat = np.array([0.5, -0.5, -0.5, 0.5]) if thumb_down else np.array([0.5, 0.5, 0.5, 0.5])
    else:
        # Object on -Y side → palm on +Y side → approach toward -Y
        approach_offset = np.array([0.0, +1.0, 0.0])
        # Palm faces -Y (toward object), THUMB UP (+Z), fingers backward (-X)
        # link6+X=world-Y, link6+Y=world+Z, link6+Z=world-X
        palm_quat = np.array([0.5, -0.5, 0.5, -0.5]) if thumb_down else np.array([0.5, 0.5, -0.5, -0.5])

    palm_pos = obj_center + approach_offset * (obj_info.obb_extents[1] + clearance)
    pre_grasp_pos = palm_pos + approach_offset * 0.10

    return palm_pos, palm_quat, pre_grasp_pos


def propose_grasps(objects: list[ObjectInfo],
                   presets: dict[str, np.ndarray] | None = None,
                   grasp_config: dict | None = None) -> list[GraspProposal]:
    """Generate ranked grasp proposals for all segmented objects.

    For each object, looks up the configured grasp types (from YAML) and
    computes a palm pose for each. Returns all proposals sorted by score.
    """
    if presets is None:
        presets = load_presets()
    if grasp_config is None:
        grasp_config = load_object_grasp_config()

    proposals = []
    for obj in objects:
        obj_grasps = grasp_config.get(obj.name, [])
        if not obj_grasps:
            continue

        for rank, grasp_def in enumerate(obj_grasps):
            gtype = grasp_def["type"]
            approach = grasp_def["approach"]
            desc = grasp_def.get("description", "")

            if gtype not in presets:
                continue

            # Per-object clearance override (banana needs more room)
            obj_clearance = CLEARANCE_OVERRIDES.get(obj.name)

            # Compute palm pose based on approach direction
            if approach == "top":
                if obj_clearance is not None:
                    palm_pos, palm_quat, pre_pos = _compute_palm_pose_top(obj, clearance=obj_clearance)
                else:
                    palm_pos, palm_quat, pre_pos = _compute_palm_pose_top(obj)
            elif approach == "side":
                if obj_clearance is not None:
                    palm_pos, palm_quat, pre_pos = _compute_palm_pose_side(obj, clearance=obj_clearance, thumb_down=(obj.name == 'cracker_box'))
                else:
                    palm_pos, palm_quat, pre_pos = _compute_palm_pose_side(obj, thumb_down=(obj.name == 'cracker_box'))
            else:
                continue

            # Score: prefer the first (preferred) grasp, penalize fallbacks
            score = 1.0 / (1.0 + rank)

            # Bonus for objects with more points (better observed = more confident)
            score *= min(1.0, obj.points.shape[0] / 500.0)

            proposals.append(GraspProposal(
                object_name=obj.name,
                grasp_type=gtype,
                approach=approach,
                description=desc,
                palm_pos=palm_pos,
                palm_quat=palm_quat,
                pre_grasp_pos=pre_pos,
                finger_angles=presets[gtype],
                score=score,
            ))

    # Sort by descending score
    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals