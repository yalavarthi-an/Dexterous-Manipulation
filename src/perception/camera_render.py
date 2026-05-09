"""
Camera rendering for the Piper+RUKA tabletop scene.

Provides:
  - load_scene()          : load the full_scene.xml and apply home keyframe
  - get_camera_info()     : extract intrinsic/extrinsic matrices for a camera
  - render_camera()       : produce RGB and metric-depth images from a camera
  - render_all_cameras()  : convenience wrapper that does all named cameras

Coordinate conventions:
  MuJoCo cameras look down their -Z axis (camera +Z points OUT of the lens,
  toward the viewer). To produce point clouds in OpenCV / Open3D convention
  (where +Z is the look direction, into the scene), we flip Y and Z when
  computing 3D points. See pointcloud.py for the deprojection math.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


# Default render dimensions — RealSense D435-like
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


@dataclasses.dataclass
class CameraInfo:
    """All the information needed to interpret a render from a given camera."""
    name: str
    width: int
    height: int
    fovy_deg: float          # vertical FOV in degrees
    K: np.ndarray            # 3x3 intrinsic matrix
    T_world_cam: np.ndarray  # 4x4 camera-to-world transform (OpenCV convention)
    near: float              # near clipping plane in meters
    far: float               # far clipping plane in meters

    def to_dict(self) -> dict:
        """For JSON serialization."""
        return {
            "name": self.name,
            "width": int(self.width),
            "height": int(self.height),
            "fovy_deg": float(self.fovy_deg),
            "K": self.K.tolist(),
            "T_world_cam": self.T_world_cam.tolist(),
            "near": float(self.near),
            "far": float(self.far),
        }


@dataclasses.dataclass
class RenderedView:
    """An RGB+depth render from a specific camera, plus its calibration."""
    info: CameraInfo
    rgb: np.ndarray          # uint8 (H, W, 3)
    depth: np.ndarray        # float32 (H, W) — distance in meters along camera Z


# ============================================================================
# Scene loading
# ============================================================================

def load_scene(xml_path: Path | str, apply_home: bool = True,
               settle_steps: int = 100) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the scene XML, apply the 'home' keyframe, and settle physics.

    Args:
        xml_path: path to the scene MJCF
        apply_home: if True, set qpos/ctrl to the 'home' keyframe before settling
        settle_steps: number of mj_step iterations to run before returning

    Returns:
        (model, data) ready for rendering. Objects have settled on the table.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if apply_home:
        kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kf >= 0:
            # Robot keyframe specifies arm + hand joints; free joints (objects)
            # keep their MJCF-default initial pose. We only overwrite the
            # robot-controlled portion of qpos.
            data.qpos[:model.nu] = model.key_qpos[kf, :model.nu]
            data.ctrl[:] = model.key_ctrl[kf]

    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    return model, data


# ============================================================================
# Camera info: intrinsics + extrinsics
# ============================================================================

def get_camera_info(model: mujoco.MjModel, data: mujoco.MjData, name: str,
                    width: int = DEFAULT_WIDTH,
                    height: int = DEFAULT_HEIGHT) -> CameraInfo:
    """Extract intrinsics and extrinsics for a named camera.

    Intrinsic matrix is computed from MuJoCo's vertical FOV (fovy):
        f_y = (height / 2) / tan(fovy / 2)
        f_x = f_y                                 (square pixels assumed)
        c_x = width / 2,  c_y = height / 2

    Extrinsic matrix is the camera's pose in the world frame, but converted
    from MuJoCo convention (looks along -Z) to OpenCV convention (looks along
    +Z) so that downstream point-cloud code is standard.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise ValueError(f"Camera '{name}' not found in model")

    # ---- Intrinsics ----
    fovy_deg = float(model.cam_fovy[cam_id])
    fovy_rad = np.deg2rad(fovy_deg)
    fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # ---- Extrinsics ----
    # data.cam_xpos and data.cam_xmat are in WORLD coordinates and reflect
    # the current kinematic state. cam_xmat is row-major (3x3 flattened).
    cam_pos = data.cam_xpos[cam_id].copy()              # (3,)
    cam_R_mujoco = data.cam_xmat[cam_id].reshape(3, 3)  # (3, 3)

    # Convert MuJoCo camera frame (looks along -Z) to OpenCV convention
    # (looks along +Z): negate the X and Z basis vectors of the camera frame
    # in world coordinates. Equivalent to a 180-degree rotation around the Y axis.
    flip = np.diag([1.0, -1.0, -1.0])  # flip Y and Z
    cam_R_opencv = cam_R_mujoco @ flip

    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_R_opencv
    T_world_cam[:3, 3]  = cam_pos

    # ---- Near/far ----
    # Use the global model defaults (per-camera near/far is rare in MuJoCo)
    near = float(model.vis.map.znear * model.stat.extent)
    far  = float(model.vis.map.zfar  * model.stat.extent)

    return CameraInfo(
        name=name,
        width=width, height=height,
        fovy_deg=fovy_deg,
        K=K, T_world_cam=T_world_cam,
        near=near, far=far,
    )


def list_cameras(model: mujoco.MjModel) -> list[str]:
    """Return the names of all cameras declared in the model."""
    names = []
    for cid in range(model.ncam):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        if nm:
            names.append(nm)
    return names


# ============================================================================
# Rendering
# ============================================================================

def render_camera(model: mujoco.MjModel, data: mujoco.MjData,
                  cam_name: str,
                  width: int = DEFAULT_WIDTH,
                  height: int = DEFAULT_HEIGHT) -> RenderedView:
    """Render RGB and metric depth from a named camera.

    Uses two passes:
      - one with depth disabled to get the colored RGB image
      - one with depth enabled to get the depth buffer
    The Renderer is reused across passes to share the GL context.
    """
    info = get_camera_info(model, data, cam_name, width, height)

    # MuJoCo's Renderer manages the offscreen GL context.
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        # ---- RGB pass ----
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render().copy()  # (H, W, 3) uint8

        # ---- Depth pass ----
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render().copy()  # (H, W) float32, distance in meters
        renderer.disable_depth_rendering()

    # MuJoCo returns metric depth directly (meters from the camera plane).
    # Pixels at the far clip return values close to `far`. We treat anything
    # farther than 99% of `far` as invalid (sky or out-of-range).
    invalid_mask = depth > 0.99 * info.far
    depth = depth.astype(np.float32)
    depth[invalid_mask] = 0.0  # 0 means "no measurement" by convention

    return RenderedView(info=info, rgb=rgb, depth=depth)


def render_all_cameras(model: mujoco.MjModel, data: mujoco.MjData,
                       cameras: Iterable[str] | None = None,
                       width: int = DEFAULT_WIDTH,
                       height: int = DEFAULT_HEIGHT) -> list[RenderedView]:
    """Render every named camera, or a subset if `cameras` is supplied."""
    cam_names = list(cameras) if cameras is not None else list_cameras(model)
    return [render_camera(model, data, n, width, height) for n in cam_names]
