"""Central configuration for Member D implementation.

Enforces reproducibility, path references, spatial constants, and model hyperparameters.
"""
from __future__ import annotations

from pathlib import Path

# Pinned random seed
SEED: int = 42

# Workspace directories (all relative to workspace root)
ROOT_DIR = Path(__file__).resolve().parent.parent
MEMBER_D_DIR = ROOT_DIR / "member_d"
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_TRAIN = DATA_DIR / "raw" / "train"
DATA_RAW_STUDENT_TEST = DATA_DIR / "raw" / "student_test"
MODELS_DIR = MEMBER_D_DIR / "models"
RESULTS_DIR = MEMBER_D_DIR / "results"
MAPS_DIR = MEMBER_D_DIR / "maps"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Geographic & Projection constants
EARTH_RADIUS_M: float = 6371000.0
DEFAULT_CROP_MARGIN_DEG: float = 0.01  # ~1.1 km margin around GPS bounding box
SAMPLE_SPACING_M: float = 15.0         # Spatial indexing point sample interval along arc

# Candidate generation parameters
CANDIDATE_SEARCH_RADIUS_M: float = 50.0
CANDIDATE_MAX_K: int = 10
CANDIDATE_MIN_COUNT: int = 3
CANDIDATE_MAX_RADIUS_M: float = 250.0

# 20 invariant feature names from shared pipeline
FEATURE_NAMES: list[str] = [
    "perp_dist",
    "dist_rank",
    "dist_minus_min",
    "rel_dist",
    "along_arc_t",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "speed",
    "dt_prev",
    "dt_next",
    "heading_diff_abs",
    "cos_heading_diff",
    "sin_heading_diff",
    "num_candidates",
    "has_reverse_twin",
    "from_node_deg",
    "to_node_deg",
    "context_mean_dist_w3",
    "context_is_nearest_w3",
]

# Random Forest Hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": SEED,
    "n_jobs": -1,
}

# HMM & Viterbi Parameters
HMM_BETA_M: float = 10.0               # Scale parameter for distance discrepancy |d_net - d_gc|
HMM_EMISSION_WEIGHT: float = 1.0       # Weight for log emission probability
HMM_EMISSION_EPSILON: float = 1e-6     # Numerical floor for emission probability
HMM_MAX_VELOCITY_MPS: float = 45.0     # ~162 km/h maximum vehicle velocity
HMM_TIME_GAP_THRESHOLD_S: float = 60.0 # Time delta above which Markov chain breaks
HMM_DISCONNECTED_LOG_PENALTY: float = -50.0 # Soft log penalty for topologically disconnected candidate pairs
HMM_UTURN_LOG_PENALTY: float = -10.0   # Penalty for switching onto immediate reverse twin
