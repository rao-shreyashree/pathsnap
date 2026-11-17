"""Unit tests for Random Forest Classifier (Phase B)."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import pytest
import numpy as np

from member_d.random_forest import RandomForestModel
from member_d.config import FEATURE_NAMES, SEED
from src.pathsnap.candidates import CandidateSet


def test_rf_model_fit_and_predict():
    # 20 features
    n_features = len(FEATURE_NAMES)
    np.random.seed(SEED)
    X = np.random.randn(100, n_features).astype(np.float32)
    y = np.random.choice([0, 1], size=100, p=[0.9, 0.1]).astype(np.int32)

    model = RandomForestModel()
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert len(probs) == 100
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_rf_score_candidates_normalization():
    model = RandomForestModel()
    np.random.seed(SEED)
    X = np.random.randn(5, len(FEATURE_NAMES)).astype(np.float32)
    y = np.array([1, 0, 0, 1, 0], dtype=np.int32)
    model.fit(X, y)

    # 2 observations: obs 0 has 3 candidates (arcs 10, 11, 12), obs 1 has 2 candidates (arcs 20, 21)
    cand_set = CandidateSet(
        obs_indices=np.array([0, 0, 0, 1, 1], dtype=np.int32),
        arc_ids=np.array([10, 11, 12, 20, 21], dtype=np.int32),
        arc_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        perp_dists=np.array([5.0, 10.0, 15.0, 2.0, 8.0], dtype=np.float64),
        along_arc_ts=np.array([0.5, 0.2, 0.8, 0.1, 0.9], dtype=np.float64),
        proj_xs=np.zeros(5),
        proj_ys=np.zeros(5),
        cand_offsets=np.array([0, 3, 5], dtype=np.int32),
    )

    cands_by_obs, scores_by_obs, top1 = model.score_candidates(X, cand_set)

    assert len(cands_by_obs) == 2
    assert cands_by_obs[0] == [10, 11, 12]
    assert cands_by_obs[1] == [20, 21]

    # Scores must sum to 1.0 for each observation
    assert pytest.approx(sum(scores_by_obs[0]), rel=1e-5) == 1.0
    assert pytest.approx(sum(scores_by_obs[1]), rel=1e-5) == 1.0

    # Top-1 must be a valid candidate from each observation
    assert top1[0] in [10, 11, 12]
    assert top1[1] in [20, 21]


def test_rf_save_and_load(tmp_path: Path):
    model = RandomForestModel()
    X = np.random.randn(20, len(FEATURE_NAMES)).astype(np.float32)
    y = np.random.choice([0, 1], size=20).astype(np.int32)
    model.fit(X, y)

    save_file = tmp_path / "test_rf.joblib"
    model.save(save_file)
    assert save_file.exists()

    loaded = RandomForestModel.load(save_file)
    probs_orig = model.predict_proba(X)
    probs_loaded = loaded.predict_proba(X)
    np.testing.assert_allclose(probs_orig, probs_loaded)
