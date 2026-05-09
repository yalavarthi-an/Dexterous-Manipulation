"""
Build-time tool: stitches the YCB source MJCFs into a single includable scene
fragment that can be <include>'d in our scene XML.

Why this exists: the YCB MJCFs in the repo are standalone documents with their
own <worldbody>. MuJoCo's <include> directive merges files but only allows one
<worldbody> per resolved model — you can't include 5 standalone MJCFs because
they'd all want to be the worldbody. The fix is to extract just the relevant
fragments (assets + body subtree, no <worldbody>) and combine them.

This script also:
  - Adjusts contact bitmasks so YCB objects collide with both Piper arm (contype=1)
    and RUKA fingers (contype=2). We use contype=3, conaffinity=3 (binary 11).
  - Replaces the placeholder inertia values (all 0.001) with rough physical estimates.
  - Re-paths mesh references to be relative to our scene's meshdir.
  - Applies a starting (pos, quat) per object so they sit on the table at startup.

Usage:
    python scripts/build_objects_fragment.py
Output:
    assets/scene/ycb_objects.xml  (the include-able fragment)

This is a build-time tool: re-run it only when you change the object set or
their starting layout. The output XML is checked in and used directly at runtime.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent

# =========================================================================
# Configuration: which objects, where they start, and physical properties
# =========================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
YCB_DIR = REPO_ROOT / "assets" / "ycb_objects"
SOURCE_MESH_DIR = REPO_ROOT / "assets" / "ycb_objects" / "meshes"
DEST_MESH_DIR = REPO_ROOT / "assets" / "mounted" / "meshes" / "ycb"
OUT_PATH = REPO_ROOT / "assets" / "scene" / "ycb_objects.xml"

# Table is at world (+0.5, 0, 0.7), 0.6m wide x 0.4m deep, surface at z=0.7
# Objects are placed on the table, with z chosen above the table surface
# so they settle naturally under gravity instead of penetrating from frame 0.
TABLE_TOP_Z = 0.70

OBJECTS = [
    # Layout v5: tall objects pushed to x=0.60 for maximum spread.
    # Short objects at x=0.30-0.45, tall objects at x=0.60.
    # All within arm's verified reach (x≤0.60, IK error 4.9mm).
    # Minimum spacing: 30cm+ between any pair.
    #
    #          Y
    #   +0.30  | mug(0.30,+0.25)                cracker_box(0.60,+0.30)
    #          |
    #    0.00  |              tennis(0.45,0.00)
    #          |
    #   -0.30  | banana(0.30,-0.25)              mustard(0.60,-0.30)
    #          └─────────────────────────────→ X
    #         0.30         0.45         0.60
    #
    {
        "name": "banana",
        "pos":  (0.30, -0.25, TABLE_TOP_Z + 0.05),
        "quat": (0.7071, 0, 0, 0.7071),
        "mass": 0.066,
        "diaginertia": (5.5e-5, 5.5e-5, 6e-6),
    },
    {
        "name": "mug",
        "pos":  (0.30,  0.25, TABLE_TOP_Z + 0.06),
        "quat": (1.0, 0, 0, 0),
        "mass": 0.118,
        "diaginertia": (1.6e-4, 1.6e-4, 1.4e-4),
    },
    {
        "name": "cracker_box",
        "pos":  (0.60,  0.30, TABLE_TOP_Z + 0.10),
        "quat": (1.0, 0, 0, 0),
        "mass": 0.411,
        "diaginertia": (1.7e-3, 1.4e-3, 6.5e-4),
    },
    {
        "name": "mustard_bottle",
        "pos":  (0.60, -0.30, TABLE_TOP_Z + 0.10),
        "quat": (1.0, 0, 0, 0),
        "mass": 0.603,
        "diaginertia": (1.1e-3, 1.1e-3, 4.5e-4),
    },
    {
        "name": "tennis_ball",
        "pos":  (0.45,  0.00, TABLE_TOP_Z + 0.04),
        "quat": (1.0, 0, 0, 0),
        "mass": 0.058,
        "diaginertia": (3.6e-5, 3.6e-5, 3.6e-5),
    },
]


# =========================================================================
# XML manipulation
# =========================================================================

def load_source_mjcf(name: str) -> ET.Element:
    """Load a YCB source MJCF and return its root."""
    path = YCB_DIR / f"{name}.xml"
    if not path.exists():
        raise FileNotFoundError(f"YCB source MJCF missing: {path}")
    return ET.parse(path).getroot()


def transform_object(name: str, cfg: dict) -> tuple[ET.Element, ET.Element]:
    """
    Read the source MJCF for `name` and return:
      (asset_fragment, body_fragment)
    Each is an ET Element whose children we will splice into our combined fragment.
    """
    root = load_source_mjcf(name)

    # ---- Extract <asset> children ----
    src_asset = root.find("asset")
    if src_asset is None:
        raise ValueError(f"{name}: no <asset> element")
    new_asset = ET.Element("asset")
    for child in src_asset:
        # Re-path mesh files: source MJCFs use paths like "banana/textured.obj".
        # We've copied YCB meshes to assets/mounted/meshes/ycb/, and the YCB fragment
        # is included from assets/scene/. With meshdir="" in the parent scene file,
        # paths in the fragment resolve relative to the fragment's own directory
        # (assets/scene/). So we need: ../mounted/meshes/ycb/banana/textured.obj.
        if child.tag in ("mesh", "texture"):
            old_file = child.get("file", "")
            if old_file:
                new_file = f"../mounted/meshes/ycb/{old_file}"
                child.set("file", new_file)
        new_asset.append(child)

    # ---- Extract the single <body> from <worldbody> ----
    src_world = root.find("worldbody")
    if src_world is None:
        raise ValueError(f"{name}: no <worldbody> element")
    bodies = list(src_world.findall("body"))
    if len(bodies) != 1:
        raise ValueError(f"{name}: expected exactly 1 body, found {len(bodies)}")
    body = bodies[0]

    # Apply starting pose
    body.set("pos", f"{cfg['pos'][0]} {cfg['pos'][1]} {cfg['pos'][2]}")
    body.set("quat", f"{cfg['quat'][0]} {cfg['quat'][1]} {cfg['quat'][2]} {cfg['quat'][3]}")
    if "euler" in body.attrib:
        del body.attrib["euler"]  # quat takes precedence; remove conflict

    # Fix contact bitmasks on collision geoms (visual geoms keep contype=0)
    for geom in body.findall("geom"):
        contype = int(geom.get("contype", "0"))
        conaff  = int(geom.get("conaffinity", "0"))
        if contype != 0 or conaff != 0:
            # Collision geom: set both bits (1 for arm, 2 for hand)
            geom.set("contype", "3")
            geom.set("conaffinity", "3")
            # Bump friction up — default contacts often slip
            geom.set("friction", "1.5 0.05 0.001")
            # Add a small contact margin so the contact solver finds contacts
            # before geometric penetration starts to bite
            geom.set("solref", "0.01 1")
            geom.set("solimp", "0.95 0.99 0.001")

    # Fix inertial: replace placeholder with our table's value
    inertial = body.find("inertial")
    if inertial is not None:
        body.remove(inertial)
    new_inertial = ET.SubElement(body, "inertial")
    new_inertial.set("pos", "0 0 0")
    new_inertial.set("mass", f"{cfg['mass']}")
    diag = cfg["diaginertia"]
    new_inertial.set("diaginertia", f"{diag[0]:.6g} {diag[1]:.6g} {diag[2]:.6g}")
    # Move <inertial> to the start of the body (MuJoCo prefers it there)
    body.remove(new_inertial)
    body.insert(0, new_inertial)

    return new_asset, body


def build_fragment() -> str:
    """Build the combined include-able fragment as a string."""
    combined_asset = ET.Element("asset")
    combined_world = ET.Element("worldbody")

    for cfg in OBJECTS:
        asset, body = transform_object(cfg["name"], cfg)
        for child in asset:
            combined_asset.append(child)
        combined_world.append(body)

    # Wrap in a top-level mujocoinclude — MuJoCo accepts this as an includable fragment
    root = ET.Element("mujocoinclude")
    # Comment header
    header = ET.Comment(
        " AUTO-GENERATED by scripts/build_objects_fragment.py — do not edit by hand. "
    )
    root.append(header)
    root.append(combined_asset)
    root.append(combined_world)

    # Pretty-print: ET.indent (Python 3.9+)
    ET.indent(root, space="  ")
    body_str = ET.tostring(root, encoding="unicode")

    # Add an XML declaration and a top comment
    return dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!--
          YCB objects fragment for the tabletop scene.
          Generated from {len(OBJECTS)} source MJCFs in assets/ycb_objects/.
          Re-run scripts/build_objects_fragment.py to regenerate after layout changes.
        -->
        """) + body_str + "\n"


def copy_ycb_meshes():
    """Copy YCB mesh trees into the unified mesh directory."""
    import shutil
    DEST_MESH_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in OBJECTS:
        name = cfg["name"]
        src = SOURCE_MESH_DIR / name
        dst = DEST_MESH_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"YCB mesh source missing: {src}")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    print(f"Copied {len(OBJECTS)} YCB mesh trees to {DEST_MESH_DIR}")


def main():
    copy_ycb_meshes()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = build_fragment()
    OUT_PATH.write_text(text)
    print(f"Wrote {OUT_PATH}")
    print(f"Object count: {len(OBJECTS)}")
    for cfg in OBJECTS:
        print(f"  {cfg['name']:18s} pos={cfg['pos']} mass={cfg['mass']:.3f}kg")


if __name__ == "__main__":
    main()