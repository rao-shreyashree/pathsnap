"""Central configuration (Members A and D).

Override the data location with env var PATHSNAP_DATA.
RADIUS_M / MAX_CAND must stay in sync with the features built by
src.build_dataset (they must match at API time): don't change them
without telling Member A.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- reproducibility
SEED: int = 42

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_ROOT                      # alias used by Member D code
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", PROJECT_ROOT / "data")).resolve()
DATA_DIR = DATA_ROOT                         # alias used by Member D code
PARTITIONS = ["train", "student_test"]       # sub-folders of DATA_ROOT

DATA_RAW_TRAIN = DATA_ROOT / "train"
DATA_RAW_STUDENT_TEST = DATA_ROOT / "student_test"
DATA_RAW_EVALUATOR_TEST = DATA_ROOT / "evaluator_test"   # held out, never fit on it

# Member A outputs
OUT_DIR = PROJECT_ROOT / "data" / "processed"
EDA_DIR = PROJECT_ROOT / "results" / "eda"
MAPS_DIR = PROJECT_ROOT / "results" / "maps"

# Member D outputs
MEMBER_D_DIR = PROJECT_ROOT / "member_d"
MODELS_DIR = MEMBER_D_DIR / "models"
RESULTS_DIR = MEMBER_D_DIR / "results"
D_MAPS_DIR = MEMBER_D_DIR / "maps"

for _folder in [OUT_DIR, EDA_DIR, MAPS_DIR, MODELS_DIR, RESULTS_DIR, D_MAPS_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- geography
EARTH_RADIUS_M: float = 6371000.0
DEFAULT_CROP_MARGIN_DEG: float = 0.01   # ~1.1 km margin around GPS bounding box
SAMPLE_SPACING_M: float = 15.0          # spatial index sample interval along an arc

# ---------------------------------------------------------------- shared candidate generation (Member A)
RADIUS_M: float = 300.0   # candidate search radius (from the EDA radius sweep)
MAX_CAND: int = 50        # keep at most this many nearest arcs per GPS point
MIN_MOVE_M: float = 3.0   # below this displacement GPS heading is unreliable

# ---------------------------------------------------------------- Member D's own candidate settings
# NOTE: these differ from RADIUS_M / MAX_CAND above. Use them only if D's HMM
# builds its own candidate set; otherwise use the shared tables.
CANDIDATE_SEARCH_RADIUS_M: float = 50.0
CANDIDATE_MAX_K: int = 10
CANDIDATE_MIN_COUNT: int = 3
CANDIDATE_MAX_RADIUS_M: float = 250.0

# ---------------------------------------------------------------- Member D feature list
# NOTE: A's shared list (16 columns) is FEATURE_COLS in src/candidates.py
# and data/processed/meta.json. This 20-name list must match it before
# training on the shared tables.
D_FEATURE_NAMES: list[str] = [
    "perp_dist", "dist_rank", "dist_minus_min", "rel_dist",
    "along_arc_t", "dist_to_start", "dist_to_end", "arc_length",
    "speed", "dt_prev", "dt_next",
    "heading_diff_abs", "cos_heading_diff", "sin_heading_diff",
    "num_candidates", "has_reverse_twin", "from_node_deg", "to_node_deg",
    "context_mean_dist_w3", "context_is_nearest_w3",
]

# ---------------------------------------------------------------- Random Forest (Member D)
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": SEED,
    "n_jobs": -1,
}

# ---------------------------------------------------------------- HMM / Viterbi (Member D)
HMM_BETA_M: float = 10.0                      # scale for |d_net - d_gc| discrepancy
HMM_EMISSION_WEIGHT: float = 1.0              # weight on log emission probability
HMM_EMISSION_EPSILON: float = 1e-6            # numerical floor for emission probability
HMM_MAX_VELOCITY_MPS: float = 45.0            # ~162 km/h
HMM_TIME_GAP_THRESHOLD_S: float = 60.0        # time gap that breaks the Markov chain
HMM_DISCONNECTED_LOG_PENALTY: float = -50.0   # soft penalty for disconnected candidate pairs
HMM_UTURN_LOG_PENALTY: float = -10.0          # penalty for switching onto the reverse twin