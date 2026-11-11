"""Camera pen pose for the OrcaHand pen-spin policy.

Image -> pen tags (multi-tag PnP) + wrist AprilTag -> pen pose in the palm frame,
in the policy's pen convention (origin at the shaft midpoint, +X toward A_top).
Quaternions are (w, x, y, z) everywhere, as in MuJoCo, mjlab and penspin_deploy.
"""

from pathlib import Path

PERCEPTION = Path(__file__).resolve().parents[1]
PEN_SPIN = PERCEPTION.parent
PEN_TAGS = PEN_SPIN / "hardware/tags/pen_tags.json"
CONFIG = PERCEPTION / "config"
