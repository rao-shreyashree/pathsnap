"""Random Forest Classifier for Road Arc Candidate Matching.

Member D — `d-rf-hmm`
Implements candidate-level scoring, group-aware cross-validation,
calibrated probability normalization, and model persistence.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure workspace root is on sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from typing import Any, Sequence
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from member_d.config import (
    DATA_RAW_TRAIN,
    FEATURE_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    RF_PARAMS,
    SEED,
)
from src.pathsnap.candidates import CandidateSet, generate_candidates
from src.pathsnap.features import extract_features_for_trip
from src.pathsnap.geo import get_centroid, project_equirectangular
from src.pathsnap.io import load_trip
from src.pathsnap.labels import derive_labels_monotone, load_route
from src.pathsnap.network import (
    build_spatial_network_index,
    compute_node_degrees,
    crop_network_to_bbox,
    find_reverse_twin_arcs,
)

QC_EXCLUDED_TRIPS = {"00000002", "00000033", "00000054"}


class RandomForestModel:
    """Wrapper around trained RandomForest model and its feature preprocessor."""

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        if pipeline is None:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(**RF_PARAMS)),
            ])
        else:
            self.pipeline = pipeline

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestModel:
        """Fit preprocessing scaler and Random Forest classifier."""
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of match (class 1) for candidate rows."""
        # Ensure class 1 probability is returned
        probs = self.pipeline.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    def score_candidates(
        self,
        X: np.ndarray,
        candidates: CandidateSet,
    ) -> tuple[list[list[int]], list[list[float]], list[int]]:
        """Score candidate arcs and normalize probabilities per observation.
        
        Returns:
          candidates_by_obs: List of candidate arc IDs per observation
          emission_scores_by_obs: Normalized candidate probabilities [0, 1]
          top1_predictions: Argmax candidate arc ID per observation
        """
        raw_probs = self.predict_proba(X)
        n_obs = len(candidates.cand_offsets) - 1

        candidates_by_obs: list[list[int]] = []
        emission_scores_by_obs: list[list[float]] = []
        top1_predictions: list[int] = []

        for i in range(n_obs):
            st = int(candidates.cand_offsets[i])
            en = int(candidates.cand_offsets[i + 1])
            k = en - st

            if k == 0:
                candidates_by_obs.append([])
                emission_scores_by_obs.append([])
                top1_predictions.append(0)
                continue

            c_arc_ids = [int(aid) for aid in candidates.arc_ids[st:en]]
            c_scores = raw_probs[st:en].astype(np.float64)

            # Normalise over candidates of this observation
            s_sum = float(np.sum(c_scores))
            if s_sum > 0:
                c_norm_scores = (c_scores / s_sum).tolist()
            else:
                c_norm_scores = [1.0 / k] * k

            best_idx = int(np.argmax(c_scores))
            best_arc = c_arc_ids[best_idx]

            candidates_by_obs.append(c_arc_ids)
            emission_scores_by_obs.append(c_norm_scores)
            top1_predictions.append(best_arc)

        return candidates_by_obs, emission_scores_by_obs, top1_predictions

    def save(self, filepath: str | Path | None = None) -> Path:
        """Save pipeline to joblib file."""
        if filepath is None:
            filepath = MODELS_DIR / "random_forest_model.joblib"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    @classmethod
    def load(cls, filepath: str | Path | None = None) -> RandomForestModel:
        """Load trained pipeline from joblib file."""
        if filepath is None:
            filepath = MODELS_DIR / "random_forest_model.joblib"
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        pipeline = joblib.load(path)
        return cls(pipeline=pipeline)


def extract_trip_dataset(
    trip_dir: str | Path,
    max_obs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, CandidateSet, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and ground-truth candidate labels for one trip.
    
    Returns:
      X: (N_cand, 20) feature array
      y: (N_cand,) binary indicator (1 if candidate matches ground truth)
      candidates: CandidateSet
      ref_arc_ids: (N_obs,) ground truth reference arc ID per observation
      px, py: (N_obs,) projected coordinates
      t_vals: (N_obs,) timestamps in seconds
    """
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    trip = load_trip(trip_dir)

    min_lon, max_lon = float(trip.track["lon"].min()), float(trip.track["lon"].max())
    min_lat, max_lat = float(trip.track["lat"].min()), float(trip.track["lat"].max())
    lon0, lat0 = get_centroid(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    px, py = project_equirectangular(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy(), lon0, lat0)
    t_vals = trip.track["t"].to_numpy()

    # Network crop and spatial index
    c_nodes, c_arcs, _ = crop_network_to_bbox(trip.nodes, trip.arcs, min_lon, max_lon, min_lat, max_lat)
    sp_idx = build_spatial_network_index(c_nodes, c_arcs, lon0, lat0)
    candidates = generate_candidates(px, py, sp_idx, radius_m=50.0, max_k=10)
    twins = find_reverse_twin_arcs(c_arcs)
    in_deg, out_deg = compute_node_degrees(len(c_nodes), c_arcs)
    X = extract_features_for_trip(trip.track, px, py, candidates, sp_idx, twins, in_deg, out_deg, c_arcs)

    # Reference route alignment
    route_file = trip_dir / f"{tid}.route"
    if route_file.exists():
        route = load_route(route_file)
        alignment = derive_labels_monotone(trip.track, trip.nodes, trip.arcs, route)
        ref_arc_ids = alignment.ref_arc_ids
        # Candidate binary label: 1 if candidate arc equals ground truth reference arc
        y = (candidates.arc_ids == ref_arc_ids[candidates.obs_indices]).astype(np.int32)
    else:
        ref_arc_ids = np.zeros(len(px), dtype=np.int32)
        y = np.zeros(len(candidates.arc_ids), dtype=np.int32)

    # Optional subsampling of observations (preserving all candidates of chosen fixes)
    if max_obs is not None and len(px) > max_obs:
        step = int(np.ceil(len(px) / max_obs))
        sampled_obs = np.arange(0, len(px), step)
        keep_mask = np.isin(candidates.obs_indices, sampled_obs)
        X = X[keep_mask]
        y = y[keep_mask]

    return X, y, candidates, ref_arc_ids, np.column_stack([px, py]), t_vals


def train_and_evaluate_rf(
    n_trips: int = 25,
    max_obs_per_trip: int = 400,
    n_splits: int = 5,
) -> tuple[RandomForestModel, dict[str, Any]]:
    """Train Random Forest with trip-aware GroupKFold and save model and metrics."""
    train_dirs = sorted([d for d in DATA_RAW_TRAIN.iterdir() if d.is_dir() and d.name not in QC_EXCLUDED_TRIPS])
    selected_dirs = train_dirs[:n_trips]

    print(f"[Phase B] Extracting features from {len(selected_dirs)} training trips...")
    t_start = time.time()

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []

    for idx, t_dir in enumerate(selected_dirs):
        X_t, y_t, _, _, _, _ = extract_trip_dataset(t_dir, max_obs=max_obs_per_trip)
        all_X.append(X_t)
        all_y.append(y_t)
        all_groups.append(np.full(len(y_t), idx, dtype=np.int32))

    X_full = np.vstack(all_X)
    y_full = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    t_extract = time.time() - t_start

    print(f"[Phase B] Extracted {len(y_full)} candidate rows ({np.sum(y_full)} positive, {np.mean(y_full)*100:.2f}%) in {t_extract:.1f}s")

    # GroupKFold Cross-Validation
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = {"accuracy": [], "precision": [], "recall": [], "macro_f1": [], "log_loss": []}

    print(f"[Phase B] Running {n_splits}-fold Group-aware CV (grouped by trip)...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_full, y_full, groups)):
        X_tr, y_tr = X_full[train_idx], y_full[train_idx]
        X_val, y_val = X_full[val_idx], y_full[val_idx]

        model = RandomForestModel()
        model.fit(X_tr, y_tr)

        val_probs = model.predict_proba(X_val)
        val_preds = (val_probs >= 0.5).astype(int)

        cv_scores["accuracy"].append(float(accuracy_score(y_val, val_preds)))
        cv_scores["precision"].append(float(precision_score(y_val, val_preds, zero_division=0)))
        cv_scores["recall"].append(float(recall_score(y_val, val_preds, zero_division=0)))
        cv_scores["macro_f1"].append(float(f1_score(y_val, val_preds, average="macro", zero_division=0)))
        cv_scores["log_loss"].append(float(log_loss(y_val, np.clip(val_probs, 1e-15, 1 - 1e-15))))

    cv_summary = {
        "macro_f1_mean": round(float(np.mean(cv_scores["macro_f1"])), 4),
        "macro_f1_std": round(float(np.std(cv_scores["macro_f1"])), 4),
        "precision_mean": round(float(np.mean(cv_scores["precision"])), 4),
        "recall_mean": round(float(np.mean(cv_scores["recall"])), 4),
        "accuracy_mean": round(float(np.mean(cv_scores["accuracy"])), 4),
        "log_loss_mean": round(float(np.mean(cv_scores["log_loss"])), 4),
        "n_samples": int(len(y_full)),
        "n_trips": int(len(selected_dirs)),
        "feature_names": FEATURE_NAMES,
    }

    print(f"[Phase B] Group CV Macro F1: {cv_summary['macro_f1_mean']} +/- {cv_summary['macro_f1_std']}")
    print(f"[Phase B] Candidate Precision: {cv_summary['precision_mean']}, Recall: {cv_summary['recall_mean']}, Log Loss: {cv_summary['log_loss_mean']}")

    # Fit final model on all data
    t_fit_start = time.time()
    final_model = RandomForestModel()
    final_model.fit(X_full, y_full)
    fit_time = time.time() - t_fit_start

    # Benchmark inference speed (per 1,000 observations / ~8,000 candidate rows)
    test_rows = X_full[:8000]
    t_inf_start = time.time()
    _ = final_model.predict_proba(test_rows)
    inf_time_ms = (time.time() - t_inf_start) * 1000.0

    meta = {
        "model_name": "RandomForestClassifier",
        "algorithm_family": "Ensemble (Bagging)",
        "rf_params": RF_PARAMS,
        "seed": SEED,
        "train_time_s": round(fit_time, 2),
        "inference_time_ms_per_1000_obs": round(inf_time_ms, 2),
        "cv_summary": cv_summary,
    }

    # Persist model and metadata
    saved_path = final_model.save()
    meta_path = RESULTS_DIR / "rf_training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Phase B] Final model saved to {saved_path}")
    print(f"[Phase B] Training metadata saved to {meta_path}")

    return final_model, meta


if __name__ == "__main__":
    train_and_evaluate_rf()
