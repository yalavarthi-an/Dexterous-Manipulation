# Diagnostic Tools

I keep four diagnostic utilities under `scripts/` that I used while assembling the model
and verifying the mount. I separated them from the main pipeline so they do not clutter
the end-to-end execution path, but I still rely on them when something looks wrong.

## `sweep_joints.py`

This script loads the combined model and sweeps every actuator through its full range, one at a
time, with the rest held at home. I use it to confirm the kinematic tree is wired correctly and
that joint limits look reasonable.

```bash
python scripts/sweep_joints.py             # interactive viewer
python scripts/sweep_joints.py --headless  # logs only, no viewer
python scripts/sweep_joints.py --duration 3.0  # seconds per joint
python scripts/sweep_joints.py --robot     # robot-only (no scene)
```

Notes:
- I use a single tightly-coupled physics+viewer loop so the viewer stays responsive
  through all 21 sweeps (~52 seconds total).
- The script falls back to joint range when the actuator's `ctrlrange` is unset.

## `verify_mount.py`

I wrote this for a three-level verification of the mount transform:

1. **Symbolic.** It parses the quaternion from the XML, builds the rotation matrix, and
   checks that the finger-direction vector aligns exactly with the tool axis.
   It reports an exact angle (typically 0.000°).
2. **Numerical.** It loads the model, runs forward kinematics at the home pose, and
   measures the actual angular deviation between fingertip directions and the tool
   axis. I expect a small number representing finger splay (~6°).
3. **Interpenetration.** It steps the sim a few times and lists any contact pairs with
   negative distance.

```bash
python scripts/verify_mount.py
```

Sample output (passing):
```
Angle between finger direction and link6 +Z (tool axis): 0.000°
  PASS
Angular deviation: 6.640°
  PASS
Active contacts at home: 0
  PASS
```

## `diagnose_mount.py`

This script reports the world-frame positions of the flange site, palm body, and all five
fingertip sites, plus a per-axis projection analysis showing which link6 axis the
fingers point along.

I use it when I am designing or debugging the mount transform — it gives me the raw numerical
ground truth that is hard to read off the viewer.

```bash
python scripts/diagnose_mount.py
```

## `check_contacts.py`

This script lists every active contact pair in the model at the home pose, with body names, geom
names, and penetration distances. It also prints the world-frame z-coordinate of every
body so I can spot anything that starts underground.

I use it to sanity-check that no two parts of the robot are clipping into each other
when stationary.

```bash
python scripts/check_contacts.py
```

## `record_all_grasps.py` (Task 5 demo)

Demo clip: [YouTube top-down grasp demo](https://youtu.be/AAiWuzB72V0).

Offline quad-mosaic recorder: two fixed cameras on the tabletop plus two lateral views riding the palm ±thumb axes. Each object run loads `full_scene.xml`, settles physics, calls `execute_best_proposal(...)`, captures RGB frames every *n*-th physics step (`--stride`), and stitches a 2×2 MP4 (requires **`imageio[ffmpeg]`** in `requirements.txt`).

```bash
# proposals path defaults to outputs/grasp_proposals.json — generate first
python scripts/visualize_grasps.py --save
python scripts/record_all_grasps.py --out outputs/demo_all_grasps.mp4
python scripts/record_all_grasps.py --objects banana,tennis_ball --stride 4
```

OpenCV is optional — it only improves on-frame captions; the recorder runs without it.
