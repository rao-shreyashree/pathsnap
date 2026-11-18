"""Central settings. Override the data location with env var PATHSNAP_DATA."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", PROJECT_ROOT / "data")).resolve()
PARTITIONS = ["train", "student_test"]   # sub-folders of DATA_ROOT

OUT_DIR = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "results" / "eda"
MAPS_DIR = PROJECT_ROOT / "results" / "maps"

for folder in [OUT_DIR, EDA_DIR, MAPS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RADIUS_M = 300.0     # candidate search radius (tune using the EDA radius sweep)
MAX_CAND = 50        # keep at most this many nearest candidate arcs per GPS point
MIN_MOVE_M = 3.0     # below this displacement the GPS heading is treated as unreliable
SEED = 42
