"""Comprehensive Evaluation and Comparison Harness for Member D.

Member D — `d-rf-hmm`
Compares:
1. Random Forest Classifier Only (Top-1 argmax)
2. Random Forest + HMM + Viterbi (Decoded optimal route)

Evaluates on identical observations and ground-truth monotone reference routes.
Outputs CSV summaries, metrics JSON, and interactive Folium HTML maps.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from member_d.config import (
    DATA_RAW_STUDENT_TEST,
    DATA_RAW_EVALUATOR_TEST,
    MAPS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)
from member_d.hmm_decoder import HMMDecoder
from member_d.random_forest import RandomForestModel
from member_d.road_graph import RoadGraph
from member_d.visualize_route import create_trip_map
from src.pathsnap.candidates import generate_candidates
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


def compute_route_continuity(pred_arcs: Sequence[int], road_graph: RoadGraph) -> tuple[float, int]:
    """Compute route continuity and count disconnected breaks.
    
    Formula:
      Among pairs (t, t+1) where pred_arcs[t] != pred_arcs[t+1],
      fraction where road_graph.is_valid_transition(pred_arcs[t], pred_arcs[t+1]) == True.
    """
    n = len(pred_arcs)
    if n <= 1:
        return 1.0, 0

    transitions = 0
    valid_transitions = 0
    breaks = 0

    for t in range(n - 1):
        a_t = int(pred_arcs[t])
        a_next = int(pred_arcs[t + 1])

        if a_t != a_next:
            transitions += 1
            if road_graph.is_valid_transition(a_t, a_next):
                valid_transitions += 1
            else:
                breaks += 1

    if transitions == 0:
        return 1.0, 0
    return float(valid_transitions / transitions), breaks


def evaluate_trip(
    trip_dir: str | Path,
    model: RandomForestModel,
    generate_map: bool = False,
) -> dict[str, Any]:
    """Run end-to-end evaluation for one trip comparing RF-only vs RF+HMM+Viterbi."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    t0_total = time.time()

    # 1. Load raw files
    t0_load = time.time()
    trip = load_trip(trip_dir)
    route_file = trip_dir / f"{tid}.route"
    has_ground_truth = route_file.exists()
    t_load = time.time() - t0_load

    # 2. Geometry projection and road graph construction
    t0_graph = time.time()
    min_lon, max_lon = float(trip.track["lon"].min()), float(trip.track["lon"].max())
    min_lat, max_lat = float(trip.track["lat"].min()), float(trip.track["lat"].max())
    lon0, lat0 = get_centroid(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    px, py = project_equirectangular(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy(), lon0, lat0)
    timestamps = trip.track["t"].to_numpy()

    # Build road graph
    road_graph = RoadGraph.from_trip_directory(trip_dir, crop=True)
    t_graph = time.time() - t0_graph

    # 3. Spatial indexing and candidate generation
    t0_cand = time.time()
    c_nodes, c_arcs, _ = crop_network_to_bbox(trip.nodes, trip.arcs, min_lon, max_lon, min_lat, max_lat)
    sp_idx = build_spatial_network_index(c_nodes, c_arcs, lon0, lat0)
    candidates = generate_candidates(px, py, sp_idx, radius_m=50.0, max_k=10)
    twins = find_reverse_twin_arcs(c_arcs)
    in_deg, out_deg = compute_node_degrees(len(c_nodes), c_arcs)
    X = extract_features_for_trip(trip.track, px, py, candidates, sp_idx, twins, in_deg, out_deg, c_arcs)
    t_cand_feat = time.time() - t0_cand

    # 4. Reference labels (if ground truth available)
    ref_arc_ids = None
    ref_route_arcs = None
    if has_ground_truth:
        ref_route_arcs = load_route(route_file)
        alignment = derive_labels_monotone(trip.track, trip.nodes, trip.arcs, ref_route_arcs)
        ref_arc_ids = alignment.ref_arc_ids

    # 5. Classifier Scoring (Random Forest Only)
    t0_rf = time.time()
    cands_by_obs, scores_by_obs, top1_rf = model.score_candidates(X, candidates)
    t_rf = time.time() - t0_rf

    # 6. HMM + Viterbi Decoding
    t0_hmm = time.time()
    decoder = HMMDecoder(road_graph=road_graph)
    hmm_result = decoder.decode(
        candidates_by_obs=cands_by_obs,
        emission_scores_by_obs=scores_by_obs,
        track_coords_xy=np.column_stack([px, py]),
        timestamps=timestamps,
    )
    top1_hmm = hmm_result["decoded_route"]
    t_hmm = time.time() - t0_hmm

    t_total = time.time() - t0_total
    n_obs = len(px)

    # 7. Compute Continuity Metrics
    cont_rf, breaks_rf = compute_route_continuity(top1_rf, road_graph)
    cont_hmm, breaks_hmm = compute_route_continuity(top1_hmm, road_graph)

    # 8. Compute Ground Truth Metrics
    top1_acc_rf = None
    top1_acc_hmm = None
    miss_rate = None
    macro_f1 = None
    precision = None
    recall = None

    if ref_arc_ids is not None:
        top1_acc_rf = float(np.mean(np.array(top1_rf) == ref_arc_ids))
        top1_acc_hmm = float(np.mean(np.array(top1_hmm) == ref_arc_ids))

        # Candidate miss rate
        misses = 0
        for i in range(n_obs):
            ref_a = ref_arc_ids[i]
            if ref_a not in cands_by_obs[i]:
                misses += 1
        miss_rate = float(misses / n_obs)

        # Candidate level F1
        y_true = (candidates.arc_ids == ref_arc_ids[candidates.obs_indices]).astype(int)
        y_probs = model.predict_proba(X)
        y_pred = (y_probs >= 0.5).astype(int)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))

    # Optional map generation
    if generate_map:
        map_path = MAPS_DIR / f"trip_{tid}.html"
        create_trip_map(
            trip_id=tid,
            track_df=trip.track,
            road_graph=road_graph,
            ref_route_arcs=ref_route_arcs,
            rf_pred_arcs=top1_rf,
            hmm_pred_arcs=top1_hmm,
            output_html_path=map_path,
        )

    return {
        "trip_id": tid,
        "n_obs": n_obs,
        "top1_acc_rf": round(top1_acc_rf, 4) if top1_acc_rf is not None else None,
        "top1_acc_hmm": round(top1_acc_hmm, 4) if top1_acc_hmm is not None else None,
        "acc_delta": round(top1_acc_hmm - top1_acc_rf, 4) if (top1_acc_rf is not None and top1_acc_hmm is not None) else None,
        "candidate_miss_rate": round(miss_rate, 4) if miss_rate is not None else None,
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "continuity_rf": round(cont_rf, 4),
        "continuity_hmm": round(cont_hmm, 4),
        "breaks_rf": breaks_rf,
        "breaks_hmm": breaks_hmm,
        "hmm_n_breaks": hmm_result["n_breaks"],
        "t_load_ms": round(t_load * 1000, 1),
        "t_graph_ms": round(t_graph * 1000, 1),
        "t_cand_feat_ms": round(t_cand_feat * 1000, 1),
        "t_rf_ms": round(t_rf * 1000, 1),
        "t_hmm_ms": round(t_hmm * 1000, 1),
        "t_total_ms": round(t_total * 1000, 1),
    }


def run_full_evaluation(model_path: str | Path | None = None) -> pd.DataFrame:
    """Run comparative evaluation across all 10 student-test development trips."""
    if model_path is None:
        model_path = MODELS_DIR / "random_forest_model.joblib"
    model = RandomForestModel.load(model_path)

    test_dirs = sorted([d for d in DATA_RAW_STUDENT_TEST.iterdir() if d.is_dir()])
    print(f"[Phase E] Evaluating {len(test_dirs)} student-test trips...")

    results: list[dict[str, Any]] = []

    for d in test_dirs:
        # Generate interactive Folium map for the first 3 trips
        gen_map = d.name in {"00000000", "00000007", "00000052"}
        print(f"[Phase E] Running Trip {d.name} (generate_map={gen_map})...")
        res = evaluate_trip(d, model=model, generate_map=gen_map)
        results.append(res)
        print(f"  -> Top-1 RF: {res['top1_acc_rf']} | Top-1 HMM: {res['top1_acc_hmm']} (Delta: {res['acc_delta']:+0.4f}) | Continuity: {res['continuity_rf']} -> {res['continuity_hmm']} | Breaks: {res['breaks_rf']} -> {res['breaks_hmm']}")

    df = pd.DataFrame(results)

    # Save detailed CSV
    csv_path = RESULTS_DIR / "student_test_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Compute overall aggregates
    summary = {
        "n_trips": int(len(df)),
        "total_observations": int(df["n_obs"].sum()),
        "mean_top1_rf": round(float(df["top1_acc_rf"].mean()), 4),
        "mean_top1_hmm": round(float(df["top1_acc_hmm"].mean()), 4),
        "mean_acc_delta": round(float(df["acc_delta"].mean()), 4),
        "mean_continuity_rf": round(float(df["continuity_rf"].mean()), 4),
        "mean_continuity_hmm": round(float(df["continuity_hmm"].mean()), 4),
        "total_breaks_rf": int(df["breaks_rf"].sum()),
        "total_breaks_hmm": int(df["breaks_hmm"].sum()),
        "mean_candidate_miss_rate": round(float(df["candidate_miss_rate"].mean()), 4),
        "mean_macro_f1": round(float(df["macro_f1"].mean()), 4),
        "mean_inference_time_ms_per_trip": round(float(df["t_total_ms"].mean()), 1),
    }

    summary_path = RESULTS_DIR / "student_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("MEMBER D EVALUATION SUMMARY (STUDENT TEST - 10 TRIPS)")
    print("="*70)
    print(f"Mean Top-1 Accuracy:  RF-Only: {summary['mean_top1_rf']:.4f}  -->  RF+HMM: {summary['mean_top1_hmm']:.4f}  (Gain: {summary['mean_acc_delta']:+0.4f})")
    print(f"Mean Route Continuity: RF-Only: {summary['mean_continuity_rf']:.4f}  -->  RF+HMM: {summary['mean_continuity_hmm']:.4f}")
    print(f"Total Disconnected Breaks: RF-Only: {summary['total_breaks_rf']}  -->  RF+HMM: {summary['total_breaks_hmm']} (Reduction: -{summary['total_breaks_rf'] - summary['total_breaks_hmm']})")
    print(f"Mean Candidate Miss Rate: {summary['mean_candidate_miss_rate']:.4f}")
    print(f"Mean Latency per Trip: {summary['mean_inference_time_ms_per_trip']:.1f} ms")
    print("="*70)
    print(f"Results CSV saved to: {csv_path}")
    print(f"Summary JSON saved to: {summary_path}")

    return df


def run_evaluator_test_evaluation(model_path: str | Path | None = None) -> pd.DataFrame:
    """Run comparative evaluation across all 10 evaluator-test trips."""
    if model_path is None:
        model_path = MODELS_DIR / "random_forest_model.joblib"
    model = RandomForestModel.load(model_path)

    if not DATA_RAW_EVALUATOR_TEST.exists():
        raise FileNotFoundError(f"Evaluator test directory not found: {DATA_RAW_EVALUATOR_TEST}")

    test_dirs = sorted([d for d in DATA_RAW_EVALUATOR_TEST.iterdir() if d.is_dir()])
    print(f"[Evaluator Test] Evaluating {len(test_dirs)} reserved evaluator trips...")

    results: list[dict[str, Any]] = []

    for d in test_dirs:
        # Generate interactive Folium map for the first 3 evaluator trips
        gen_map = d.name in {"00000015", "00000023", "00000073"}
        print(f"[Evaluator Test] Running Trip {d.name} (generate_map={gen_map})...")
        res = evaluate_trip(d, model=model, generate_map=gen_map)
        results.append(res)
        print(f"  -> Top-1 RF: {res['top1_acc_rf']} | Top-1 HMM: {res['top1_acc_hmm']} (Delta: {res['acc_delta']:+0.4f}) | Continuity: {res['continuity_rf']} -> {res['continuity_hmm']} | Breaks: {res['breaks_rf']} -> {res['breaks_hmm']}")

    df = pd.DataFrame(results)

    # Save detailed CSV
    csv_path = RESULTS_DIR / "evaluator_test_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Compute overall aggregates
    summary = {
        "n_trips": int(len(df)),
        "total_observations": int(df["n_obs"].sum()),
        "mean_top1_rf": round(float(df["top1_acc_rf"].mean()), 4),
        "mean_top1_hmm": round(float(df["top1_acc_hmm"].mean()), 4),
        "mean_acc_delta": round(float(df["acc_delta"].mean()), 4),
        "mean_continuity_rf": round(float(df["continuity_rf"].mean()), 4),
        "mean_continuity_hmm": round(float(df["continuity_hmm"].mean()), 4),
        "total_breaks_rf": int(df["breaks_rf"].sum()),
        "total_breaks_hmm": int(df["breaks_hmm"].sum()),
        "mean_candidate_miss_rate": round(float(df["candidate_miss_rate"].mean()), 4),
        "mean_macro_f1": round(float(df["macro_f1"].mean()), 4),
        "mean_inference_time_ms_per_trip": round(float(df["t_total_ms"].mean()), 1),
    }

    summary_path = RESULTS_DIR / "evaluator_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("MEMBER D EVALUATION SUMMARY (EVALUATOR TEST - 10 TRIPS)")
    print("="*70)
    print(f"Mean Top-1 Accuracy:  RF-Only: {summary['mean_top1_rf']:.4f}  -->  RF+HMM: {summary['mean_top1_hmm']:.4f}  (Gain: {summary['mean_acc_delta']:+0.4f})")
    print(f"Mean Route Continuity: RF-Only: {summary['mean_continuity_rf']:.4f}  -->  RF+HMM: {summary['mean_continuity_hmm']:.4f}")
    print(f"Total Disconnected Breaks: RF-Only: {summary['total_breaks_rf']}  -->  RF+HMM: {summary['total_breaks_hmm']} (Reduction: -{summary['total_breaks_rf'] - summary['total_breaks_hmm']})")
    print(f"Mean Candidate Miss Rate: {summary['mean_candidate_miss_rate']:.4f}")
    print(f"Mean Latency per Trip: {summary['mean_inference_time_ms_per_trip']:.1f} ms")
    print("="*70)
    print(f"Results CSV saved to: {csv_path}")
    print(f"Summary JSON saved to: {summary_path}")

    return df


if __name__ == "__main__":
    run_full_evaluation()
    if DATA_RAW_EVALUATOR_TEST.exists():
        run_evaluator_test_evaluation()
