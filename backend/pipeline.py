import time
import logging
import joblib
import numpy as np
from pathlib import Path

from src.data_io import load_trip
from src.candidates import build_trip_tables, FEATURE_COLS

log = logging.getLogger("pathsnap")

MODEL_PATH = Path("models/artifacts/boosting_tuned.joblib")
EVAL_DIR = Path("data/evaluator_test")      # TODO: match the real zip layout
PARTITION = "evaluator_test"
USE_HMM = True

try:
    from src.hmm import decode_route        # TODO: D's final import path
except ImportError:
    decode_route = None


class Pipeline:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.tracks = {p.stem: p for p in sorted(EVAL_DIR.glob("*.track"))}
        log.info("loaded model, %d trips", len(self.tracks))

    def trip_ids(self):
        return list(self.tracks)

    def _score(self, trip):
        trip.route = None                    # labels never reach the model
        obs, cand = build_trip_tables(trip)
        cand = cand.reset_index(drop=True)
        cand["prob"] = self.model.predict_proba(cand[FEATURE_COLS])[:, 1]
        return trip, obs, cand

    def predict(self, trip_id: str) -> dict:
        t0 = time.time()
        path = self.tracks[trip_id]
        trip, obs, cand = self._score(load_trip(path, PARTITION))

        # classifier-only: argmax per GPS point
        top = cand.loc[cand.groupby("obs_idx")["prob"].idxmax()].sort_values("obs_idx")
        clf_arcs = dict(zip(top["obs_idx"], top["arc_id"]))

        method, hmm_arcs, warnings = "classifier_only", None, []
        if USE_HMM and decode_route is not None:
            groups = cand.sort_values(["obs_idx"]).groupby("obs_idx", sort=True)
            obs_ids = list(groups.groups.keys())
            candidates = [g["arc_id"].tolist() for _, g in groups]
            scores = [g["prob"].tolist() for _, g in groups]
            ts = obs.set_index("obs_idx").loc[obs_ids, "t"].tolist() if "t" in obs else None
            out = decode_route(candidates, scores, trip.nodes, trip.arcs, ts)
            warnings = out.get("warnings", [])
            if out.get("success"):
                hmm_arcs = dict(zip(obs_ids, out["decoded_route"]))  # ASSUMES 1 arc per obs
                method = "hmm_viterbi"
        elif USE_HMM:
            warnings.append("decoder not available; classifier-only used")

        final = hmm_arcs or clf_arcs
        metrics = self._evaluate(path, clf_arcs, hmm_arcs, trip.arcs)
        log.info("trip=%s method=%s metrics=%s time=%.2fs",
                 trip_id, method, metrics, time.time() - t0)
        return {
            "trip_id": trip_id, "method": method, "warnings": warnings,
            "route_arc_ids": [int(a) for a in dict.fromkeys(final.values())],
            "route_geometry": self._geometry(trip, final),
            "gps_points": [[float(la), float(lo)] for lo, la in
                           zip(trip.track[:, 0], trip.track[:, 1])],  # TODO: match Trip attrs
            "metrics": metrics,
            "elapsed_seconds": round(time.time() - t0, 3),
        }

    def _geometry(self, trip, arcs_by_obs):
        seq = list(dict.fromkeys(arcs_by_obs.values()))
        out = []
        for a in seq:
            u, v = trip.arcs[a]
            out.append([[float(trip.nodes[u][1]), float(trip.nodes[u][0])],
                        [float(trip.nodes[v][1]), float(trip.nodes[v][0])]])
        return out

    def _evaluate(self, path, clf_arcs, hmm_arcs, arcs):
        # second copy WITH route, only used server-side
        gt_trip = load_trip(path, PARTITION)
        obs_gt, _ = build_trip_tables(gt_trip)
        ref = dict(zip(obs_gt["obs_idx"], obs_gt["ref_arc"]))
        valid = [k for k, v in ref.items() if v is not None and v >= 0]

        def acc(pred):
            return float(np.mean([pred.get(k) == ref[k] for k in valid])) if valid else None

        def continuity(pred):
            seq = [pred[k] for k in sorted(pred)]
            ok = sum(a == b or arcs[a][1] == arcs[b][0] for a, b in zip(seq, seq[1:]))
            return ok / max(len(seq) - 1, 1)

        m = {"classifier_top1": acc(clf_arcs),
             "classifier_continuity": continuity(clf_arcs),
             "n_points": len(valid)}
        if hmm_arcs:
            m |= {"hmm_top1": acc(hmm_arcs), "hmm_continuity": continuity(hmm_arcs)}
        return m