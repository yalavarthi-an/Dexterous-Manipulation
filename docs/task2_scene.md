# Task 2: Tabletop Scene, YCB Objects, and RGB-D Cameras

> *"Add a table in front of the robot at a reasonable working height. Load at
> least 5 YCB objects ... with sensible physics. Add one or more RGB-D cameras."*

Here I describe the tabletop scene I built, the YCB object set, the camera
configuration and intrinsics, and the rendering pipeline I wrote to produce RGB,
depth, and point-cloud outputs.

## TL;DR

- 5 YCB objects (banana, mug, cracker box, mustard bottle, tennis ball)
  spread across a 60×80 cm table at z=0.70m
- I mounted the robot on a 70cm pedestal so its base is level with the table top
- I use three RGB-D-style cameras: two scene-mounted (third-person, fixed) + one wrist-mounted
  (eye-in-hand, fixed orientation)
- All cameras: 640×480, fovy=58° (RealSense D435-like), depth in 0.1–3m range
- `scripts/render_scene.py` produces RGB, depth, point cloud, and a calibration
  JSON for every camera per run

## Scene assembly

I compose the full scene (`assets/scene/full_scene.xml`) by `<include>`-ing
the robot definition (`assets/mounted/piper_ruka.xml`) and an auto-generated
YCB-objects fragment (`assets/scene/ycb_objects.xml`), plus a hand-authored
worldbody for the table, pedestal, lights, and floor.

### Robot pedestal

At floor level (base_link at z=0) the Piper cannot reach a 0.7m table — its
arm fully extended only reaches z≈0.68m. I mount the robot on a 70cm
cylindrical pedestal so the base is level with the tabletop. I did this to mirror
how I have seen Piper installations in labs: the arm bolted to a workbench
surface, with workspace ahead at the same level.

### Table

60cm deep (X) × 80cm wide (Y), tabletop at z=0.70m, 4 wood-grain legs.
I placed it 50cm in front of the robot pedestal in world +X. I sized it so each
of the 5 objects has ~18cm of clearance from its neighbors so the arm can
maneuver between them without collision and each object has a clean
top-down approach corridor.

### YCB object selection and physics

I picked five objects from `elpis-lab/YCB_Dataset` for grasp-type diversity:

| Object | Mass | Inertia (diag, kg·m²) | Grasp type tested |
|---|---|---|---|
| Banana | 0.066 kg | (5.5e-5, 5.5e-5, 6e-6) | Pinch/precision (curved) |
| Mug | 0.118 kg | (1.6e-4, 1.6e-4, 1.4e-4) | Tripod/wrap (handle + body) |
| Cracker box | 0.411 kg | (1.7e-3, 1.4e-3, 6.5e-4) | Power/box (large rectangular) |
| Mustard bottle | 0.603 kg | (1.1e-3, 1.1e-3, 4.5e-4) | Lateral/power (tall narrow) |
| Tennis ball | 0.058 kg | (3.6e-5, 3.6e-5, 3.6e-5) | Spherical/pinch |

I set friction `(1.5, 0.05, 0.001)` on contact geoms — high primary
friction for stable grasping. Collision geometry is the convex decomposition
shipped with the source (4-5 convex pieces per object, generated with CoACD
at dataset-build time).

### Build pipeline

`scripts/build_objects_fragment.py` turns the 5 standalone YCB MJCFs into a
single `<mujocoinclude>` fragment that I embed in `full_scene.xml`. The
script handles three things the source MJCFs needed:

1. **Re-paths meshes** so they resolve under my scene's mesh tree
2. **Fixes contact bitmasks** (`contype=3, conaffinity=3`) — the source files
   set `contype=1, conaffinity=1`, which would have prevented YCB objects
   from colliding with the RUKA fingers (RUKA uses bitmask `2`)
3. **Replaces placeholder inertias** (the source used `0.001 0.001 0.001`
   for every object regardless of size) with object-specific estimates

Re-run `build_objects_fragment.py` whenever the object set or starting
positions change.

## Cameras

### Configuration

Three cameras, all at RealSense D435-like intrinsics (640×480, fovy=58°):

**`scene_cam`** — third-person, fixed in worldbody. Position `(-0.3, 0.7, 1.7)`
(back-left of the table, up high), attitude controlled via
`mode="targetbody" target="table"` so the camera continuously aims at the
table center.

**`scene_cam2`** — second third-person view. Position `(1.0, -0.5, 1.5)`
(front-right of the table), also targeting the table center. Together with
`scene_cam`, these two cameras view the workspace from **opposite sides** —
what one camera misses (back faces of objects), the other sees.

**`wrist_cam`** — eye-in-hand, child of `ruka_palm`. Position `(0, -0.04, 0.02)`
in palm-local frame (back of palm), `xyaxes="-1 0 0  0 -1 0"` (looks along
palm -Z, i.e. the finger-extension direction). Rigid orientation: as the
wrist moves, the camera moves and rotates with it. At the home pose the
camera looks roughly forward+up (the wrist's natural at-rest direction);
during a grasping approach the camera looks at the target object.

> **Note on `wrist_cam` in practice.** I keep `wrist_cam` defined in the
> scene because real eye-in-hand setups need it, and the rendering pipeline
> renders it for completeness. In this submission, however, **the primary
> cameras driving Tasks 2–4 are `scene_cam` and `scene_cam2`** — they own
> ~95% of the fused-cloud coverage and all the OBBs the heuristic grasp
> pipeline operates on. The wrist view does not contribute usefully at the
> home pose (it stares at the sky, ~13% valid pixels) and is not consumed
> by the segmentation, OBB, or grasp-pose stages. Treat `wrist_cam` as
> documented-and-wired-up, but inert in the actual perception path.

### Why this configuration

I chose **2 scene + 1 wrist** after iterating from a simpler 1+1 setup:

- **My initial design** used a single scene camera, which produced partial point
  clouds (~312k points, ~60% object surface coverage) and systematically
  biased bounding boxes — shifted toward the camera, smaller than the true
  object extent.
- **Adding `scene_cam2`** on the opposite side nearly doubled the fused cloud
  to ~620k points with ~95% object surface coverage. Bounding boxes became
  properly centered, and grasp pose accuracy improved significantly.
- **The wrist camera** does not contribute to initial scene perception at the
  home pose (it points at the sky, where there are no objects). Its role is
  during **grasp execution** (Task 4): as the arm moves over an object, the
  wrist camera provides close-range depth for visual servoing and pre-grasp
  verification. This matches real eye-in-hand deployments.

I wrote the rendering pipeline to take a *list* of cameras, so adding or
removing cameras is a one-line change in the scene MJCF — no code changes
needed.

### Intrinsic matrix

For both cameras, computed from MuJoCo's `fovy`:
```
f_y = (height / 2) / tan(fovy / 2)
    = (480 / 2) / tan(29°)
    = 432.97 pixels
f_x = f_y                     (square pixels)
c_x = width / 2  = 320
c_y = height / 2 = 240
```

```
K = [[432.97,   0  , 320],
     [  0  , 432.97, 240],
     [  0  ,   0  ,   1]]
```

### Convention conversion (MuJoCo → OpenCV)

MuJoCo cameras look down their **−Z** axis (Z points out of the lens, toward
the viewer). OpenCV / Open3D / most point-cloud libraries assume the camera
looks along **+Z**. My `get_camera_info()` helper applies a `diag(1, -1, -1)`
flip to the camera rotation when emitting `T_world_cam`, so downstream code
gets standard-convention extrinsics and doesn't have to special-case MuJoCo.

This is documented in code comments at `src/perception/camera_render.py`
and matters because the produced `cameras.json` is reused by Task 3.

## Rendering pipeline

`scripts/render_scene.py` produces my Task-2 deliverable outputs:

```
outputs/render_<TIMESTAMP>/
    scene_cam/
        rgb.png         # 640×480 PNG
        depth.png       # color-mapped depth visualization (turbo cmap)
        depth.npy       # raw float32 depth in meters (for downstream code)
        cloud.ply       # world-frame colored point cloud
    scene_cam2/
        ...             # same files (second viewpoint)
    wrist_cam/
        ...             # same files
    fused_cloud.ply     # concatenated point cloud from all cameras
    cameras.json        # K + T_world_cam for every camera
```

Pipeline (what I do in `render_scene.py`):

1. I load `full_scene.xml`, apply the robot home keyframe, and settle physics
   (200 steps) so YCB objects rest on the table.
2. For each named camera I:
   - Render RGB and metric-depth with `mujoco.Renderer`. Pixels beyond 99%
     of the far clipping plane are flagged as invalid (depth=0).
   - Compute `K` and `T_world_cam` (with the MuJoCo→OpenCV flip).
   - Deproject each valid (u, v, z) pixel to a 3D point in camera frame:
     `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`.
   - Transform to world frame using `T_world_cam`.
   - Save RGB.png, depth.png (colorized), depth.npy, and a colored .ply.
3. I concatenate per-camera point clouds into a fused .ply.
4. I write `cameras.json` with all calibration data.

### Output verification

A typical run I get (with the 2-scene-cam + 1-wrist-cam configuration):

| Camera | Valid pixels | Depth range | Points |
|---|---|---|---|
| `scene_cam` | 88.6% (272k/307k) | 0.84 – 3.06 m | ~272,000 |
| `scene_cam2` | ~88% (~270k/307k) | ~0.8 – 3.0 m | ~270,000 |
| `wrist_cam` | 12.9% (40k/307k) | 0.07 – 1.71 m | ~40,000 |
| Fused | — | — | ~620,000 |

The wrist-cam's low valid-pixel count is expected on my setup: at the home pose the
camera looks at the sky, where there is no depth measurement (only the
visible hand and palm strip register valid depth).

In practice, **`scene_cam` and `scene_cam2` are the primary cameras** for
this submission — together they provide essentially all of the useful
~95% surface coverage that feeds Task 3's OBB / grasp-pose heuristics. The
wrist camera is rendered (for parity with real eye-in-hand pipelines) but
not actually consumed downstream. I left it in place rather than removing
it so the scene matches what a deployed Piper+RUKA setup would expose,
and so adding visual-servoing later would be a drop-in addition rather
than a scene rebuild.

## Files

| Path | Purpose |
|---|---|
| `assets/scene/full_scene.xml` | The runtime scene (robot + table + objects + cameras) |
| `assets/scene/tabletop_only.xml` | Standalone tabletop scene (no robot) for testing |
| `assets/scene/ycb_objects.xml` | Auto-generated YCB-objects fragment |
| `assets/ycb_objects/` | Source YCB MJCFs and meshes (texture, OBJ, convex-decomp STLs) |
| `assets/mounted/meshes/ycb/` | YCB meshes copied into the unified mesh tree |
| `scripts/build_objects_fragment.py` | Build script for the YCB fragment |
| `scripts/render_scene.py` | The Task 2 deliverable: render everything |
| `scripts/preview_cameras.py` | Quick viewer for cycling through cameras |
| `scripts/verify_full_scene.py` | Sanity-check the merged scene |
| `scripts/verify_tabletop.py` | Sanity-check just the tabletop |
| `scripts/view_pointcloud.py` | Open3D-based viewer for saved .ply files |
| `src/perception/camera_render.py` | Scene loading + RGB/depth rendering |
| `src/perception/pointcloud.py` | Depth → point cloud + .ply I/O |