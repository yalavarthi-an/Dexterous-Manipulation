"""
Measure the bounding extent of the RUKA palm STL so we can pick the right
mount offset analytically rather than by guessing.
"""

from pathlib import Path
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
PALM_STL = REPO_ROOT / "assets" / "mounted" / "meshes" / "ruka" / "Palm.STL"

mesh = trimesh.load_mesh(str(PALM_STL))
print(f"Palm STL: {PALM_STL.name}")
print(f"  Vertex count: {len(mesh.vertices)}")
print(f"  Bounding box (min):  {mesh.bounds[0]}")
print(f"  Bounding box (max):  {mesh.bounds[1]}")
print(f"  Extent (size):       {mesh.extents}")
print(f"  Centroid:            {mesh.centroid}")

# Specifically: how far does the palm extend in +Z and -Z from origin?
# (RUKA's local +Z is the wrist direction, -Z is fingers direction.)
z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
print()
print(f"  Palm extends along its local Z axis from z={z_min:+.4f} to z={z_max:+.4f}")
print(f"    ({z_max:+.4f} = furthest +Z point, i.e., the BACK of the palm — wrist side)")
print(f"    ({z_min:+.4f} = furthest -Z point, i.e., the front edge near fingers)")
print()
print(f"  After mount transform (180° around (1,1,0)/√2), palm +Z gets flipped to link6 -Z.")
print(f"  So the back of the palm (palm +Z = {z_max:+.4f}) ends up at link6 z = -{z_max:+.4f}")
print(f"  relative to palm origin.")
print()
print(f"  For the back of the palm to sit flush at link6 z = 0 (the flange face),")
print(f"  we need: palm_origin_z = {z_max:+.4f}")
print(f"  Currently using pos=\"0 0 0.06\" which is {0.06 - z_max:+.4f}m off.")
