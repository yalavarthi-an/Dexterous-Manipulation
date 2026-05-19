# Task 3: 3D Vision-Based Grasp Prediction

## Approach Choice: Heuristic Geometric Pipeline

I chose a heuristic / analytical pipeline over a learned model
(AnyDexGrasp, UniDexGrasp, DexGraspNet). My justification:

| Criterion | Heuristic Pipeline | Learned Model |
|---|---|---|
| Hand topology match | Direct — designed for RUKA 15-DOF | Requires retargeting from Shadow/Allegro/MANO |
| Verifiability | Each step inspectable and debuggable | Black-box inference, hard to diagnose |
| Dependencies | NumPy, Open3D | MinkowskiEngine, PointNet2, CUDA, large checkpoints |
| Speed | <100ms per scene | 0.5–5s per scene |
| Deployment complexity | Zero — runs inside MuJoCo process | Requires separate inference server |

A well-justified heuristic that produces reliable grasps on RUKA is more
valuable than a misapplied learned model with unknown retargeting fidelity.
I iteratively validated the pipeline in simulation with measured IK errors
and physical success/failure feedback.

---

## Pipeline Overview

```
Scene point cloud (RGB-D cameras)
        │
        ▼
Object segmentation (known MuJoCo positions + radius crop)
        │
        ▼
OBB computation per object (PCA on point cluster)
        │
        ▼
Grasp config lookup (YAML: type + approach per object)
        │
        ▼
Palm pose computation (top-down or side approach)
        │
        ▼
Finger preset lookup (YAML: 15 joint angles per grasp type)
        │
        ▼
Ranked GraspProposal list (scored by rank + point count)
```

---

## Step 1: Point Cloud Generation

I render RGB + depth from three cameras:

| Camera | Position | Purpose |
|---|---|---|
| `scene_cam` | (−0.3, 0.7, 1.7), targets table | **Primary** overview (left-back) |
| `scene_cam2` | (1.0, −0.5, 1.5), targets table | **Primary** secondary view (right-front) |
| `wrist_cam` | Palm body, pos=(0,−0.04,0.02) | Wired up for parity with real eye-in-hand setups; **not used by the perception path** in this submission |

In practice, the heuristic pipeline below is driven entirely by
`scene_cam` + `scene_cam2`. `wrist_cam` is rendered for completeness but
contributes almost no valid depth at the home pose (~13% of pixels — it
looks at the sky there), and the segmentation / OBB / grasp-pose stages
do not benefit from it. I keep it defined so adding a visual-servoing
stage later would be a drop-in change.

I unproject depth images using pinhole camera intrinsics (from `fovy` and
image resolution). I fuse all three clouds into a single scene cloud
(~610k points) by transforming from camera frame to world frame using
`data.cam_xpos` and `data.cam_xmat`.

---

## Step 2: Object Segmentation

I segment objects using **known world-frame positions from MuJoCo**
(`data.xpos[body_id]`). For each object, I keep points within a crop radius and
above the table surface (Z > 0.71m):

| Object | Crop Radius |
|---|---|
| banana | 12cm |
| mug | 10cm |
| cracker_box | 18cm |
| mustard_bottle | 14cm |
| tennis_ball | 6cm |

---

## Step 3: OBB Computation (PCA)

For each point cluster, I compute an Oriented Bounding Box via PCA:

1. Compute centroid, center points
2. Eigen-decompose covariance matrix → 3 principal axes, sorted by descending
   eigenvalue (longest axis first)
3. Project points onto each axis, compute full extent (max − min)
4. Compute OBB center from centroid + projection midpoint offsets
5. Ensure right-handed coordinate system

**OBB outputs used downstream:**
- `obb_center`: IK target reference point
- `obb_extents[1]`: Y-axis half-extent for side approach offset
- vertical extent: Z-axis projection for top-down grasp height

---

## Step 4: Grasp Configuration (YAML)

`configs/grasp_presets.yaml` is where I define the object-to-grasp mapping:

```yaml
object_grasps:
  banana:         [precision_pinch top, power_wrap side]
  mug:            [power_wrap top, power_wrap side]
  cracker_box:    [power_wrap side, top_grasp top]
  mustard_bottle: [power_wrap side, top_grasp top]
  tennis_ball:    [spherical top, tripod top]
```

Primary grasp scores 1.0; fallbacks score 0.5, 0.33, etc.
Scores are multiplied by `min(1.0, N_points / 500)`.

---

## Step 5: Palm Pose Computation

### Reference Point Design

I placed the IK target site `palm` at `link6 pos=(0, 0, 0.18)` — the **knuckle
line** (middle/ring MCP joints), 8cm past the palm body origin. I chose that over
the palm center (z=0.15) because contact occurs at the fingers, not the palm.

### Top-Down Approach

```
Orientation:  link6+X→world−Z (palm down), link6+Z→world+X (fingers forward)
Quaternion:   (0.7071, 0, 0.7071, 0)
Palm Z:       object_top_Z + clearance  (uses OBB axis most aligned with Z)
```

**Per-object clearance overrides:**

| Object | Clearance | Reason |
|---|---|---|
| default | 2.25cm | Standard gap |
| banana | 3.0cm | Curved surface, extra room |
| mug | −1.0cm | Knuckle 1cm below mug top for insert |

### Side Approach

```
Smart direction: always approach from Y side closest to Y=0
  Object at Y>0 → approach from −Y
  Object at Y<0 → approach from +Y

Palm XY:  OBB center ± (obb_extents[1] + clearance) in approach direction
Palm Z:   OBB center Z (object mid-height)
Clearance: 2.25cm default, 1.0cm for cracker_box and mustard_bottle
Pre-grasp: palm_pos + approach_direction × 10cm
```

**Side orientation variants:**

| Object | Thumb | Quaternion | Reason |
|---|---|---|---|
| cracker_box (Y<0, approach +Y) | **DOWN** (−Z) | (0.5, −0.5, 0.5, −0.5) | Thumb-up fails at X=0.60; pronation more reachable |
| mustard_bottle (Y>0, approach −Y) | **UP** (+Z) | (0.5, 0.5, 0.5, 0.5) | Thumb-up works at Y=+0.30 |
| mug fallback (Y>0, approach −Y) | **UP** (+Z) | (0.5, 0.5, 0.5, 0.5) | X=0.30, well within reach |
| banana fallback (Y<0, approach +Y) | **UP** (+Z) | (0.5, 0.5, −0.5, −0.5) | Close to robot, thumb-up works |

I determined thumb direction empirically via `test_ik.py`. Thumb-down
(wrist pronation) is more achievable than thumb-up (supination) at full
reach (X > 0.55m). Thumb-up is better at moderate reach (X < 0.45m).

---

## Step 6: Finger Presets

| Preset | Description | Objects |
|---|---|---|
| `power_wrap` | All fingers + thumb close firmly | cracker_box, mustard_bottle, mug |
| `precision_pinch` | Index + thumb; ring/pinky fully curled | banana |
| `spherical` | All fingers spread and close around sphere | tennis_ball |
| `tripod` | Index, middle, thumb 3-point grip | tennis_ball fallback |
| `top_grasp` | Fingers curl down from above | cracker_box/mustard fallbacks |
| `open` | All joints at 0 | home/rest |

**Mug two-stage preset (in executor, not YAML):**

The mug required a sequence I could not express as a single YAML preset:

- **Function A (insert):** middle + ring MCP=0.4, DIP/PIP=0.1 (narrow claw);
  index + pinky fully curled; thumb not opposed (CMC=0)
- **Function B (grip):** middle + ring fully closed (1.5/1.5/1.5); index +
  pinky stay curled; thumb fully opposed (−1.5/1.0/1.0)

I hardcoded it in the executor because YAML supports only single-state (open → close),
not multi-stage sequences.

---

## Step 7: Reachability Filtering

I pre-check each proposal with a 6D pose IK solve at the pre-grasp
position. I skip proposals that fail this check.

**IK test results for my current layout:**

| Object | Type | App | Pose IK | Status |
|---|---|---|---|---|
| banana | precision_pinch | top | 51.9mm / 19.5° | ✅ |
| mug | power_wrap | top | 60.2mm / 17.3° | ✅ |
| cracker_box | power_wrap | side | 4.2mm / 13.9° | ✅ |
| mustard_bottle | power_wrap | side | 4.5mm / 13.9° | ✅ |
| tennis_ball | spherical | top | 5.2mm / 10.2° | ✅ |

All 5 primary grasps pass for me. 9/10 total proposals (including fallbacks) pass.

---

## Grasp Visualization

`scripts/visualize_grasps.py --save` shows (when I run it):

- **Green wireframes:** object OBBs from PCA
- **Colored axis frames:** palm poses at grasp position (RGB = XYZ axes)
- **Orange arrows:** approach trajectories (pre-grasp → grasp)
- **Magenta sphere:** palm site reference point on robot

I save proposals to `outputs/grasp_proposals.json`.
