# Task 1: Mounting the RUKA Hand on the Piper Arm

> *"This is the foundational task — every later task depends on it."*

Here I describe how I mounted the RUKA 5-finger hand on the AgileX Piper 6-DoF
arm to form a unified 21-DoF robot, the design choices I made, and how I verified
the result.

## TL;DR

I define the combined robot in `assets/mounted/piper_ruka.xml`. I re-parented RUKA's `Palm_Link` onto Piper's `link6` body via:

```xml
<body name="ruka_palm" pos="0 0 0.10" quat="0 0.707 0.707 0" childclass="ruka">
```

I confirm the mount with three independent checks (symbolic, numerical, interpenetration) in `scripts/verify_mount.py`.

## Source assets

| Robot | Source | Notes |
|---|---|---|
| Piper arm | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper) | Pre-made MJCF including 84 decomposed visual + collision meshes |
| RUKA hand | NYU RUKA repo, `assets/xml/hand_assembly.xml` | Pre-made MJCF + 16 STL meshes (palm + 5×3 finger phalanges) |

Using two pre-existing MJCFs let me avoid building a hand model from raw CAD/STLs. I had to edit both files before I could combine them.

## What had to be removed from each source

### From the Piper MJCF
- `<body name="link7">` and `<body name="link8">` — the default 2-finger gripper
- `<equality joint1="joint8" joint2="joint7" .../>` — coupled the two gripper fingers
- `<position name="gripper" .../>` actuator
- `<light spotlight target="link8" .../>` — the targeted light (re-targeted in scene file)

### From the RUKA MJCF
- `<option gravity="0 -9.81 0"/>` — RUKA was authored with **Y-up convention**
  (gravity in −Y), but Piper uses MuJoCo's default Z-up convention. Combining them
  required adopting a single convention — I kept Z-up because it's MuJoCo's default and
  matches the rest of robotics tooling.
- The `world_frame` wrapper body — RUKA's MJCF wraps everything in a dummy body that
  served as the standalone-hand world anchor; I replaced this with link6 as the parent.
- `<joint name="Palm_Joint" type="slide" .../>` — a degenerate slide joint with range
  `[-0.00001, 0]`, present in the source but unused.

## Frame-convention bridging

The two source MJCFs use different palm/flange conventions:

**Piper `link6` (the parent body):**
- +Z = approach axis (where a tool sticks out of the flange)

**RUKA `Palm_Link` (the child body):**
- +X = thumb side of the palm
- +Y = palm-out direction (away from the back of the hand)
- −Z = direction the fingers extend

Three things had to be aligned:
1. **Finger axis must equal tool axis** (RUKA −Z must align with link6 +Z)
2. **Palm-out direction should be sensible at the home pose** (I chose palm facing
   "down" toward the workspace at Piper's home pose, so a top-down grasp is the natural
   default)
3. **No interpenetration** between the back of the palm and link6's wrist housing

## Mount transform

```xml
<body name="ruka_palm" pos="0 0 0.10" quat="0 0.707 0.707 0" childclass="ruka">
```

### `pos="0 0 0.10"` — 10 cm along link6 +Z

Empirically tuned. Smaller values (0.05, 0.06) caused the back-flange of the RUKA palm
STL to penetrate link6's wrist housing — the palm origin sits in front of a substantial
amount of palm geometry that extends backwards. 0.10 m gives a small visible gap that
plausibly represents a notional wrist adapter plate.

### `quat="0 0.707 0.707 0"` — 180° rotation around (1,1,0)/√2 in link6's frame

This single rotation simultaneously achieves:

| RUKA axis | Mapped to link6 axis | Meaning |
|---|---|---|
| RUKA −Z (fingers extend) | link6 +Z | Fingers along the tool axis ✅ |
| RUKA +Y (palm-out) | link6 +X | Palm faces "down" at home pose |
| RUKA +X (thumb side) | link6 +Y | Thumb is positioned "above" the four fingers |

I preferred this orientation over alternatives like `quat="0 1 0 0"` (which is also
kinematically valid) because it places the palm in a workspace-facing orientation at the
Piper's home pose, saving joint-6 range for fine-tuning grasp orientation rather than
burning it on coarse alignment.

## Verification

Three independent levels, all run by `scripts/verify_mount.py`:

### Level 1 — Symbolic (pure math from the quaternion)

My script parses the quaternion straight from the XML, builds the rotation matrix, and
reports the angle between the finger direction and link6's +Z axis. The result is
**0.000°** — which is mathematically exact, not approximate.

### Level 2 — Numerical (forward kinematics at home)

The script loads the model, runs forward kinematics at the home keyframe, and measures
the world-frame angle between the average fingertip-from-palm direction and link6's +Z
axis. I get **6.640°** — entirely accounted for by anatomical finger splay (the
four non-thumb fingers don't all point in exactly the same direction; same as on a human
hand).

### Level 3 — Interpenetration

The script steps the simulation 20 times from home and lists active contact pairs with
negative distance. I see **0 contacts at home pose** — the model is contact-clean
when stationary.

Sample output:
```
LEVEL 1: SYMBOLIC VERIFICATION
  Angle between finger direction and link6 +Z (tool axis): 0.000°
    PASS: Fingers are aligned with the tool axis (within 1°).

LEVEL 2: NUMERICAL VERIFICATION
  Average finger direction (world):  [+0.969, -0.007, -0.247]
  link6 +Z direction        (world): [+0.991,  0.000, -0.134]
  Angular deviation: 6.640°
    PASS

LEVEL 3: INTERPENETRATION CHECK
  Active contacts at home: 0
    PASS
```

## Joint sweep verification

`scripts/sweep_joints.py` exercises every actuator through its full range so I can confirm
the kinematic tree is wired correctly. All 21 joints (6 arm + 15 hand) move
independently as I expect. The sweep also serves as a basic stress test of joint limits
and contact stability.

A note on what I observed: during certain wide arm-joint sweeps, the wrist briefly
penetrates the floor. That is a sweep-only artifact (the script commands joints to
their hard mathematical limits without regard to workspace constraints), not a model
defect — I prevent that in execution with IK and motion-planning constraints from Task 4.

## Simplifications and trade-offs

- **No physical wrist-adapter geometry.** The brief explicitly allows this:
  *"You may do this purely in the MJCF by aligning frames (no physical CAD needed)."*
  The visual gap between link6 and the palm in the rendered model represents the space
  where a real adapter plate would sit.

- **Inertia of the palm is synthesized**, not derived from the RUKA STL. The original
  RUKA MJCF uses `<compiler settotalmass="1"/>`, which renormalizes all inertias to a
  total of 1 kg — useful for relative dynamics but not physically meaningful. I replaced
  this with an explicit `<inertial>` tag for the palm body using a reasonable value
  (mass = 0.15 kg, diaginertia ~ 0.0002 kg·m²). The finger phalanx inertias were
  retained from the source.

- **No actuator-coupling for tendon-driven underactuation.** The real RUKA is a
  tendon-driven underactuated hand: 15 joints driven by ~5 motors via mechanical
  coupling. The MJCF has 15 independent actuators, treating each joint as fully
  controllable. This is a standard simplification for sim-based learning/grasping
  workflows and matches what the source RUKA MJCF does.

- **Contact friction values were retained** from each source (Piper: defaults; RUKA:
  `friction="1.5 0.01 0.001"` for fingertips). I tuned these further during Task 4 when
  contact stability mattered for grasping.

## Files

| Path | Purpose |
|---|---|
| `assets/mounted/piper_ruka.xml` | The combined robot MJCF |
| `assets/mounted/piper_ruka_scene.xml` | Minimal scene wrapper for testing |
| `assets/mounted/meshes/piper/` | Piper meshes (copied from Menagerie) |
| `assets/mounted/meshes/ruka/` | RUKA meshes (copied from NYU repo) |
| `scripts/sweep_joints.py` | Sweeps all 21 actuators through their ranges |
| `scripts/verify_mount.py` | 3-level mount correctness check |
| `scripts/diagnose_mount.py` | World-frame position and axis report |
| `scripts/check_contacts.py` | Lists active contacts at any pose |
