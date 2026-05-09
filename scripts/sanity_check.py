"""Day 1 sanity check: verify MuJoCo installs, loads, and renders."""
import mujoco
import mujoco.viewer
import numpy as np

# Minimal MJCF: a single sphere falling onto a plane
XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.5">
      <joint type="free"/>
      <geom name="ball" type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print(f"nq (generalized coords): {model.nq}")
print(f"nv (DoF): {model.nv}")
print(f"nbody: {model.nbody}")

# Launch viewer — close window when done
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
