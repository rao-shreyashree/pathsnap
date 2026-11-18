"""Evaluator test trips for the API (Laptop 1). Default folder: <repo>/data/evaluator_test (override: env PATHSNAP_EVAL).
    from src.evaluator_data import list_eval_trips, load_eval_trip
    ids = list_eval_trips()                  # for GET /trips
    trip = load_eval_trip(ids[0])            # for POST /predict (trip.route = reference arcs, keep it server-side)
Never fit, tune or pick thresholds on this data."""
import os
from functools import lru_cache
from pathlib import Path

from . import config
from .data_io import load_trip

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL", config.PROJECT_ROOT / "data" / "evaluator_test"))


@lru_cache(maxsize=1)
def _index():
    tracks = sorted(EVAL_DIR.rglob("*.track"))
    if not tracks:
        raise FileNotFoundError(f"No .track files under {EVAL_DIR}")
    stems = [t.stem for t in tracks]
    dup = {s for s in stems if stems.count(s) > 1}
    return {(t.stem if t.stem not in dup else f"{t.parent.name}_{t.stem}"): t for t in tracks}


def list_eval_trips():
    return list(_index())


def load_eval_trip(trip_id: str):
    idx = _index()
    if trip_id not in idx:
        raise KeyError(f"unknown trip {trip_id}")
    return load_trip(idx[trip_id], "evaluator_test", trip_id)
