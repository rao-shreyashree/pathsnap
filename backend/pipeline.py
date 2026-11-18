"""Full prediction pipeline: raw trip -> candidates -> model scores -> (HMM) -> route + metrics.

Ground truth: the model path only ever sees a trip with trip.route = None. A SECOND copy
of the trip (with its route) is loaded afterwards, only to score the final prediction.
"""
import logging
import time

import joblib
import pandas as pd

from backend import config as cfg
from backend.geo import arcs_to_geojson, continuity, pointwise_score
from src.candidates import FEATURE_COLS, build_trip_tables
from src.data_io import load_trip

log = logging.getLogger("pathsnap")

try:                                   # D's decoder. TODO: confirm the final import path.
    from src.hmm import decode_route
except ImportError:
    decode_route = None


def _obs_ids(obs):
    return obs["obs_idx"] if "obs_idx" in obs.columns else obs.index.to_series()


def _timestamps(obs, ids):
    for name in ("t", "time", "timestamp", "ts"):
        if name in obs.columns:
            col = obs.set_index(_obs_ids(obs))[name]
            return col.loc[ids].tolist()
    return None


class Pipeline:
    def __init__(self):
        self.model = joblib.load(cfg.MODEL_PATH)
        self.pos = list(self.model.classes_).index(1)
        self.tracks = {p.stem: p for p in sorted(cfg.TRACK_DIR.rglob("*.track"))}
        if not self.tracks:
            raise RuntimeError(f"No .track files found under {cfg.TRACK_DIR}")
        log.info("model=%s trips=%d partition=%s hmm=%s", cfg.MODEL_PATH.name,
                 len(self.tracks), cfg.PARTITION, cfg.USE_HMM and decode_route is not None)

    def trip_ids(self):
        return list(self.tracks)

    # ---- prediction (no labels anywhere in here) --------------------------------------
    def _score(self, trip_id):
        trip = load_trip(self.tracks[trip_id], cfg.PARTITION)
        trip.route = None
        obs, cand = build_trip_tables(trip)
        cand = cand.reset_index(drop=True)
        if cand.empty:
            raise ValueError("no candidate arcs were generated for this trip")
        cand["prob"] = self.model.predict_proba(cand[FEATURE_COLS])[:, self.pos]
        return trip, obs, cand

    @staticmethod
    def _classifier_only(cand):
        top = cand.loc[cand.groupby("obs_idx")["prob"].idxmax()]
        return {int(o): int(a) for o, a in zip(top["obs_idx"], top["arc_id"])}

    def _hmm(self, trip, obs, cand, warnings):
        if not cfg.USE_HMM:
            return None
        if decode_route is None:
            warnings.append("HMM decoder not available; classifier-only route returned")
            return None
        groups = cand.sort_values("obs_idx", kind="stable").groupby("obs_idx", sort=True)
        ids = [int(i) for i in groups.groups.keys()]
        try:
            out = decode_route([g["arc_id"].tolist() for _, g in groups],
                               [g["prob"].tolist() for _, g in groups],
                               trip.nodes, trip.arcs, _timestamps(obs, ids))
        except Exception:
            log.exception("decoder crashed")
            warnings.append("HMM decoder raised an error; classifier-only route returned")
            return None
        warnings.extend(out.get("warnings", []))
        route = out.get("decoded_route") or []
        if not out.get("success") or len(route) != len(ids):
            warnings.append("HMM output unusable (success=False or not one arc per observation)")
            return None
        return {i: int(a) for i, a in zip(ids, route)}

    # ---- server-side evaluation -------------------------------------------------------
    def _reference(self, trip_id):
        gt = load_trip(self.tracks[trip_id], cfg.PARTITION)       # route kept, server only
        obs_gt, _ = build_trip_tables(gt)
        return dict(zip((int(i) for i in _obs_ids(obs_gt)), obs_gt["ref_arc"]))

    def predict(self, trip_id):
        t0 = time.time()
        warnings = []
        trip, obs, cand = self._score(trip_id)
        clf = self._classifier_only(cand)
        hmm = self._hmm(trip, obs, cand, warnings)
        final = hmm if hmm is not None else clf

        ref = self._reference(trip_id)
        correct, scored = pointwise_score(final, ref)
        if scored == 0:
            warnings.append("no scoreable observations")
        c_clf, _ = pointwise_score(clf, ref)
        c_hmm = pointwise_score(hmm, ref)[0] if hmm is not None else None

        route = [final[k] for k in sorted(final)]
        elapsed = round(time.time() - t0, 3)
        log.info("predict trip=%s method=%s acc=%d/%d clf_acc=%d/%d cont=%s time=%.2fs",
                 trip_id, "hmm_viterbi" if hmm is not None else "classifier_only",
                 correct, scored, c_clf, scored, continuity(final, trip.arcs), elapsed)
        return {
            "trip_id": trip_id,
            "predicted_route": arcs_to_geojson(route, trip.nodes, trip.arcs),
            "metrics": {"pointwise_accuracy": correct / scored if scored else 0.0,
                        "correct_observations": correct, "scored_observations": scored},
            "details": {
                "method": "hmm_viterbi" if hmm is not None else "classifier_only",
                "classifier_only_accuracy": c_clf / scored if scored else None,
                "hmm_accuracy": (c_hmm / scored if scored else None) if hmm is not None else None,
                "n_observations": len(final), "n_candidate_rows": len(cand),
                "elapsed_seconds": elapsed, "warnings": warnings},
        }
