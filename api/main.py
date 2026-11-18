"""PathSnap API (Laptop 1). Contract: ui/API_CONTRACT.md
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Env: PATHSNAP_EVAL (evaluator trips dir), PATHSNAP_MODEL (RF .joblib path).
Ground truth (.route) is used server-side for metrics only and is never returned."""
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# D's package is `member_d` on branch d-rf-hmm; fall back to `src` if the merge flattened it.
try:
    from member_d.config import DATA_RAW_EVALUATOR_TEST, MODELS_DIR
    from member_d.evaluate_d import compute_route_continuity
    from member_d.hmm_decoder import HMMDecoder
    from member_d.random_forest import RandomForestModel
    from member_d.road_graph import RoadGraph
except ImportError:
    from src.config import DATA_RAW_EVALUATOR_TEST, MODELS_DIR
    from src.evaluate_d import compute_route_continuity
    from src.hmm_decoder import HMMDecoder
    from src.random_forest import RandomForestModel
    from src.road_graph import RoadGraph

from src.pathsnap.candidates import generate_candidates
from src.pathsnap.features import extract_features_for_trip
from src.pathsnap.geo import get_centroid, project_equirectangular
from src.pathsnap.io import load_trip
from src.pathsnap.labels import derive_labels_monotone, load_route
from src.pathsnap.network import (build_spatial_network_index, compute_node_degrees,
                                  crop_network_to_bbox, find_reverse_twin_arcs)

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL", DATA_RAW_EVALUATOR_TEST))
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", Path(MODELS_DIR) / "random_forest_model.joblib"))

app = FastAPI(title="PathSnap API")
_model = None
_cache: dict = {}


def _trip_dirs():
    if not EVAL_DIR.exists():
        raise HTTPException(500, f"Evaluator trip folder not found: {EVAL_DIR}")
    return {d.name: d for d in sorted(EVAL_DIR.iterdir()) if d.is_dir()}


def _get_model():
    global _model
    if _model is None:
        _model = RandomForestModel.load(MODEL_PATH)
    return _model


def _run(trip_dir: Path) -> dict:
    t0 = time.time()
    tid = trip_dir.name
    trip = load_trip(trip_dir)
    tr = trip.track
    lon0, lat0 = get_centroid(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    px, py = project_equirectangular(tr["lon"].to_numpy(), tr["lat"].to_numpy(), lon0, lat0)

    graph = RoadGraph.from_trip_directory(trip_dir, crop=True)
    c_nodes, c_arcs, _ = crop_network_to_bbox(trip.nodes, trip.arcs, float(tr["lon"].min()), float(tr["lon"].max()),
                                              float(tr["lat"].min()), float(tr["lat"].max()))
    sp = build_spatial_network_index(c_nodes, c_arcs, lon0, lat0)
    cands = generate_candidates(px, py, sp, radius_m=50.0, max_k=10)      # same settings as evaluate_d.evaluate_trip
    twins = find_reverse_twin_arcs(c_arcs)
    in_deg, out_deg = compute_node_degrees(len(c_nodes), c_arcs)
    X = extract_features_for_trip(tr, px, py, cands, sp, twins, in_deg, out_deg, c_arcs)

    cands_by_obs, scores_by_obs, top1_rf = _get_model().score_candidates(X, cands)
    hmm = HMMDecoder(road_graph=graph).decode(candidates_by_obs=cands_by_obs, emission_scores_by_obs=scores_by_obs,
                                              track_coords_xy=np.column_stack([px, py]), timestamps=tr["t"].to_numpy())
    top1_hmm = hmm["decoded_route"]
    infer_ms = round((time.time() - t0) * 1000, 1)

    metrics = {"n_obs": int(len(px)),
               "route_continuity": round(compute_route_continuity(top1_hmm, graph)[0], 4),
               "inference_ms": infer_ms}
    route_file = trip_dir / f"{tid}.route"
    if route_file.exists():                                              # server-side only
        ref = derive_labels_monotone(tr, trip.nodes, trip.arcs, load_route(route_file)).ref_arc_ids
        metrics["top1_classifier_only"] = round(float(np.mean(np.array(top1_rf) == ref)), 4)
        metrics["top1_with_hmm"] = round(float(np.mean(np.array(top1_hmm) == ref)), 4)

    segs, prev = [], None
    for a in top1_hmm:                                                   # one segment per arc, in order, no repeats
        a = int(a)
        if a == prev or not graph.is_valid_arc(a):
            continue
        g = graph.get_arc_geometry(a)
        segs.append([[g["start_lat"], g["start_lon"]], [g["end_lat"], g["end_lon"]]])
        prev = a

    gps = [{"obs_idx": i, "lat": float(la), "lon": float(lo), "ts": float(t)}
           for i, (la, lo, t) in enumerate(zip(tr["lat"], tr["lon"], tr["t"]))]
    return {"trip_id": tid, "gps": gps, "predicted_route": segs, "metrics": metrics}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/trips")
def trips():
    return {"trips": list(_trip_dirs())}


@app.post("/predict")
def predict(body: dict):
    tid = body.get("trip_id")
    dirs = _trip_dirs()
    if tid not in dirs:
        raise HTTPException(404, f"unknown trip {tid}")
    try:
        if tid not in _cache:
            _cache[tid] = _run(dirs[tid])
        return _cache[tid]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
