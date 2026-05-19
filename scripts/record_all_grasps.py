#!/usr/bin/env python3
"""
Batch-record grasp executions to a single MP4 with a 2x2 mosaic:
  [ static top           | static front (world-fixed eye + look-at) ]
  [ +thumb side (dynamic)| −thumb side (dynamic) ]

Top and front share fixed XY framing on the tabletop; the front camera uses the
same eye Z as the top camera. The lateral pair still rides the palm (+X thumb /
−X opposite). The top panel is vertically flipped after render because the passive
FoV raster often reads upside-down; pass --no-top-flip if yours does not.

Depends on imageio[ffmpeg] for encoding.

Usage:
  conda activate pathon
  cd /path/to/pathon_takehome_test
  python scripts/record_all_grasps.py --out outputs/demo_all_grasps.mp4

  python scripts/record_all_grasps.py --objects banana,tennis_ball \\
      --proposals outputs/grasp_proposals.json --stride 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio  # noqa: E402

from src.grasping.grasp_proposal import GraspProposal  # noqa: E402
from src.perception.camera_render import load_scene  # noqa: E402
from src.planning.executor import execute_best_proposal  # noqa: E402

SCENE_XML = REPO_ROOT / "assets" / "scene" / "full_scene.xml"
DEFAULT_PROPOSALS = REPO_ROOT / "outputs" / "grasp_proposals.json"

# Table body in full_scene.xml: <body name="table" pos="0.5 0 0">,
# tabletop box size 0.40 x 0.55 (half-extents), top face z ≈ 0.70 m.
TABLE_XY = np.array([0.50, 0.0], dtype=np.float64)
TABLE_SURFACE_Z = 0.70


def proposals_from_json(path: Path) -> list[GraspProposal]:
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


def _palm_kinematics(model: mujoco.MjModel, data: mujoco.MjData):
    """World-frame palm origin, finger-extension unit vector, thumb-side unit vector."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ruka_palm")
    if bid < 0:
        raise RuntimeError("Body 'ruka_palm' not found")
    R = data.xmat[bid].reshape(3, 3)
    palm_pos = np.array(data.xpos[bid], dtype=np.float64).copy()
    # R maps local column vectors to world; fingertips extend along local -Z.
    finger_w = -(R[:, 2]).astype(np.float64)
    fn = np.linalg.norm(finger_w)
    if fn < 1e-9:
        finger_w = np.array([0.0, 0.0, -1.0])
    else:
        finger_w /= fn
    thumb_w = R[:, 0].astype(np.float64)
    tn = np.linalg.norm(thumb_w)
    if tn < 1e-9:
        thumb_w = np.array([1.0, 0.0, 0.0])
    else:
        thumb_w /= tn
    return palm_pos, finger_w, thumb_w


def _sphere_camera(
    cam: mujoco.MjvCamera,
    eye: np.ndarray,
    target: np.ndarray,
    model: mujoco.MjModel,
) -> None:
    """Configure FREE camera from explicit world-frame eye and focal target."""
    delta = np.asarray(eye, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    r = float(np.linalg.norm(delta))
    if r < 1e-6:
        delta = np.array([0.0, 1e-4, 1.0])
        r = float(np.linalg.norm(delta))
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.lookat[:] = target
    cam.distance = r
    xy = delta[:2]
    horiz = float(np.hypot(xy[0], xy[1]))
    if horiz < 1e-8:
        azim = 0.0
    else:
        azim = float(np.rad2deg(np.arctan2(delta[1], delta[0])))
    elev = float(np.rad2deg(np.arctan2(delta[2], horiz)))
    cam.azimuth = azim
    cam.elevation = elev


class QuadMosaicRecorder:
    """Offscreen MuJoCo Renderer: static top/front + palm-tracked ±thumb sides."""

    def __init__(
        self,
        model: mujoco.MjModel,
        panel_w: int,
        panel_h: int,
        fovy_deg: float,
        static_focus_xyz: tuple[float, float, float],
        top_height: float,
        top_off_xy: tuple[float, float],
        static_front_y_offset: float,
        static_front_x_offset: float,
        dist_side: float,
        side_lookat_blend: float,
        flip_top_vertical: bool = True,
    ):
        self.model = model
        model.vis.global_.fovy = float(fovy_deg)
        self.renderer = mujoco.Renderer(model, height=panel_h, width=panel_w)
        self.scene_option = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.flip_top_vertical = bool(flip_top_vertical)

        self.static_tgt = np.array(static_focus_xyz, dtype=np.float64)
        oh = float(top_height)
        ox, oy = float(top_off_xy[0]), float(top_off_xy[1])
        # Static top-down: eye above focus (tiny XY jitter avoids singular spherical azimuth).
        jitter = np.array([1e-4, -1e-4, oh], dtype=np.float64)
        self.static_top_eye = self.static_tgt + jitter + np.array([ox, oy, 0.0], dtype=np.float64)

        shared_eye_z = float(self.static_top_eye[2])
        # Static front: −Y hemisphere; eye Z matches top camera exactly (same height above scene).
        self.static_front_eye = np.array(
            [
                self.static_tgt[0] + float(static_front_x_offset),
                self.static_tgt[1] - float(static_front_y_offset),
                shared_eye_z,
            ],
            dtype=np.float64,
        )

        self.dist_side = float(dist_side)
        self.side_lookat_blend = float(side_lookat_blend)

    def dynamic_side_focus(self, data: mujoco.MjData) -> np.ndarray:
        """Blend palm with table landmark so ±thumb rails stay framed on the grasp."""
        palm_pos, _, _ = _palm_kinematics(self.model, data)
        tbl = np.array([TABLE_XY[0], TABLE_XY[1], TABLE_SURFACE_Z + 0.05], dtype=np.float64)
        return self.side_lookat_blend * palm_pos + (1.0 - self.side_lookat_blend) * tbl

    def render_panel(self, data: mujoco.MjData, mode: str) -> np.ndarray:
        _, _, thumb_w = _palm_kinematics(self.model, data)

        if mode == "top":
            eye = self.static_top_eye
            tgt = self.static_tgt
        elif mode == "front":
            eye = self.static_front_eye
            tgt = self.static_tgt
        elif mode == "side_thumb":
            tgt = self.dynamic_side_focus(data)
            eye = tgt + thumb_w * self.dist_side
        elif mode == "side_opp":
            tgt = self.dynamic_side_focus(data)
            eye = tgt - thumb_w * self.dist_side
        else:
            raise ValueError(mode)

        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        _sphere_camera(self.cam, eye, tgt, self.model)
        self.renderer.update_scene(
            data,
            camera=self.cam,
            scene_option=self.scene_option,
        )
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8)
        if mode == "top" and self.flip_top_vertical:
            rgb = np.ascontiguousarray(np.flipud(rgb))
        return rgb

    def render_quad(self, data: mujoco.MjData) -> np.ndarray:
        top = self.render_panel(data, "top")
        front = self.render_panel(data, "front")
        s1 = self.render_panel(data, "side_thumb")
        s2 = self.render_panel(data, "side_opp")
        row0 = np.hstack([top, front])
        row1 = np.hstack([s1, s2])
        return np.vstack([row0, row1])


def _annotate_quad(img_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return img_rgb
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    for line in lines:
        cv2.putText(bgr, line, (6, y), font, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (6, y), font, 0.55, (24, 24, 24), 1, cv2.LINE_AA)
        y += 22
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _corner_labels(panel_h: int, panel_w: int) -> np.ndarray:
    """Return RGBA overlay with quadrant titles (semi-transparent stripe)."""
    h = int(panel_h * 2)
    w = int(panel_w * 2)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    texts = [
        (4, panel_h // 8, "Top (static)"),
        (panel_w + 4, panel_h // 8, "Front (static)"),
        (4, panel_h + panel_h // 10, "Side 1 (+thumb axis)"),
        (panel_w + 4, panel_h + panel_h // 10, "Side 2 (\u2212thumb axis)"),
    ]
    try:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x, y, txt in texts:
            cv2.putText(overlay, txt, (x, y), font, 0.55,
                        (255, 255, 255, 255), 2, cv2.LINE_AA)
    except ImportError:
        pass
    return overlay


def _blend_overlay(base: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    a = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = overlay_rgba[:, :, :3].astype(np.float32)
    b = base.astype(np.float32)
    out = rgb * a + b * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "all_grasps_quad.mp4")
    ap.add_argument("--proposals", type=Path, default=None)
    ap.add_argument(
        "--objects",
        default="banana,mug,cracker_box,mustard_bottle,tennis_ball",
        help="Comma-separated YCB bodies to run (MJCF names)",
    )
    ap.add_argument("--settle", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30.0, help="Output video frame rate")
    ap.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Emit every nth simulation frame (speeds encode, shortens playback vs sim time)",
    )
    ap.add_argument("--panel-w", type=int, default=428)
    ap.add_argument("--panel-h", type=int, default=361)
    ap.add_argument("--fovy", type=float, default=42.0, help="Visualizer vertical FoV (deg)")
    ap.add_argument(
        "--static-focus-x",
        type=float,
        default=float(TABLE_XY[0]),
        help="Shared world-frame focal point X for static top + front cameras",
    )
    ap.add_argument("--static-focus-y", type=float, default=float(TABLE_XY[1]))
    ap.add_argument(
        "--static-focus-z",
        type=float,
        default=float(TABLE_SURFACE_Z + 0.04),
        help="Typically ~surface + small offset toward scene center",
    )
    ap.add_argument(
        "--top-height",
        type=float,
        default=1.62,
        help="Static top camera: meters above focal point along +world Z (tuned to fit "
        "full tabletop at default FoV)",
    )
    ap.add_argument("--top-off-x", type=float, default=0.0)
    ap.add_argument("--top-off-y", type=float, default=0.0)
    ap.add_argument(
        "--static-front-y-offset",
        type=float,
        default=1.14,
        help="Static front eye: focal_y minus this (\u2212Y default hemisphere; larger = "
        "farther back, more tabletop in frame)",
    )
    ap.add_argument(
        "--no-top-flip",
        action="store_true",
        help="Disable vertical flip correction on the top panel if it looks upright already",
    )
    ap.add_argument(
        "--static-front-x-offset",
        type=float,
        default=0.0,
        help="Sideways displacement of static front eye along world +X vs focal point",
    )
    ap.add_argument(
        "--dist-side",
        type=float,
        default=0.88,
        help="±thumb lateral cameras only: distance along \u00b1thumb axis",
    )
    ap.add_argument(
        "--side-lookat-blend",
        type=float,
        default=0.28,
        help="\u00b1thumb rails only: 0=fixed-on-table landmark, 1=follow palm",
    )
    ap.add_argument("--quiet-proposals", action="store_true", help="Silence executor prints")
    return ap.parse_args()


def main():
    args = parse_args()
    objs = [o.strip() for o in args.objects.split(",") if o.strip()]
    prop_path = args.proposals or DEFAULT_PROPOSALS
    if not prop_path.is_file():
        print(f"No proposals JSON at {prop_path}. Generate with:\n"
              f"  python scripts/visualize_grasps.py --save")
        sys.exit(1)
    proposals = proposals_from_json(prop_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    panel_h, panel_w = int(args.panel_h), int(args.panel_w)
    label_overlay_rgba = _corner_labels(panel_h, panel_w)

    writer_kw = dict(
        fps=float(args.fps),
        codec="libx264",
        quality=8,
        ffmpeg_log_level="error",
        macro_block_size=1,
    )
    writer = imageio.get_writer(str(args.out), **writer_kw)
    sep_frames = max(6, int(args.fps * 0.6))

    verbose_exec = not args.quiet_proposals

    try:
        for idx, obj in enumerate(objs):
            caption = (
                f"Object : {obj}   |   clip {idx + 1}/{len(objs)}   |"
                "   Static top | Static front | Side1 (+thumb) | Side2 (−thumb)"
            )
            print(f"\n[{idx + 1}/{len(objs)}] Recording '{obj}' ...")

            print(f"  Loading & settling ({args.settle} steps)...")
            model, data = load_scene(SCENE_XML, settle_steps=args.settle)
            recorder = QuadMosaicRecorder(
                model,
                panel_w,
                panel_h,
                args.fovy,
                (
                    args.static_focus_x,
                    args.static_focus_y,
                    args.static_focus_z,
                ),
                args.top_height,
                (args.top_off_x, args.top_off_y),
                args.static_front_y_offset,
                args.static_front_x_offset,
                args.dist_side,
                args.side_lookat_blend,
                flip_top_vertical=(not args.no_top_flip),
            )

            step_counter = {"i": 0}

            def on_step_capture(m: mujoco.MjModel, d: mujoco.MjData) -> None:
                step_counter["i"] += 1
                if args.stride > 1 and step_counter["i"] % args.stride != 0:
                    return
                quad_frame = recorder.render_quad(d)
                quad_frame = _blend_overlay(quad_frame, label_overlay_rgba)
                quad_frame = _annotate_quad(quad_frame, [caption])
                writer.append_data(quad_frame)

            result = execute_best_proposal(
                model,
                data,
                proposals,
                obj,
                record_log=False,
                viewer=None,
                on_step=on_step_capture,
                verbose=verbose_exec,
            )

            ok: bool | None
            if result is None:
                print(f"  FAIL: no reachable proposal for '{obj}'.")
                ok = False
            else:
                ok = bool(result.success)
                print(f"  Done: success={result.success}"
                      + ("" if ok else f" ({result.failure_mode})"))

            status = "[SKIP]" if result is None else ("[OK]" if ok else "[FAIL]")
            tail_caption = f"{caption}   {status}"
            quad_hold = recorder.render_quad(data)
            quad_hold = _blend_overlay(quad_hold, label_overlay_rgba)
            quad_hold = _annotate_quad(quad_hold, [tail_caption])
            for _ in range(sep_frames):
                writer.append_data(quad_hold)

    finally:
        writer.close()
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
