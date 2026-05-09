"""
Grasp presets loader.

Reads the YAML config and provides lookup functions used by the grasp
proposal pipeline and the execution pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRESETS_YAML = REPO_ROOT / "configs" / "grasp_presets.yaml"

# Joint names in actuator order (actuators 6..20)
HAND_JOINT_NAMES = [
    "index_mcp",  "index_dip",  "index_pip",
    "middle_mcp", "middle_dip", "middle_pip",
    "ring_mcp",   "ring_dip",   "ring_pip",
    "pinky_mcp",  "pinky_dip",  "pinky_pip",
    "thumb_cmc",  "thumb_mcp",  "thumb_ip",
]

ARM_ACTUATOR_COUNT = 6    # first 6 actuators are Piper arm
HAND_ACTUATOR_COUNT = 15  # actuators 6..20 are RUKA


def load_presets(yaml_path: Path | str | None = None) -> dict[str, np.ndarray]:
    """Load grasp presets from YAML -> {name: 15-element float64 array}."""
    path = Path(yaml_path) if yaml_path else PRESETS_YAML
    with open(path) as f:
        raw = yaml.safe_load(f)

    presets = {}
    for name, joints in raw.items():
        if name == "object_grasps":
            continue
        if not isinstance(joints, dict):
            continue
        arr = np.array([joints[k] for k in HAND_JOINT_NAMES], dtype=np.float64)
        presets[name] = arr
    return presets


def load_object_grasp_config(yaml_path: Path | str | None = None) -> dict:
    """Load the object_grasps section -> {object_name: [list of grasp dicts]}."""
    path = Path(yaml_path) if yaml_path else PRESETS_YAML
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw.get("object_grasps", {})


def get_preset(name: str, presets: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Get a single preset by name. Loads from disk if presets not provided."""
    if presets is None:
        presets = load_presets()
    if name not in presets:
        raise KeyError(f"Preset '{name}' not found. Available: {list(presets.keys())}")
    return presets[name].copy()
