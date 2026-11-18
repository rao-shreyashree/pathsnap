"""Projection, reference-arc alignment, candidate generation and features.

Pipeline per trip:
  1. project everything to metres (UTM)
  2. align the reference route to GPS points (monotone DP) -> reference arc per observation
  3. for each observation, candidates = arcs within RADIUS_M (nearest MAX_CAND kept)
  4. features per (observation, candidate arc); label = candidate is the reference arc
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from . import config
from .data_io import Trip
from .geo import make_projector, bearing_deg, angdiff

FEATURE_COLS = [
    "dist_m", "dist_rank", "dist_minus_min", "n_cand", "is_nearest",
    "proj_frac", "arc_len_m", "dist_to_end_m",
    "heading_diff", "cos_heading_diff", "heading_valid",
    "arc_bearing_sin", "arc_bearing_cos",
    "speed_mps", "dt_prev", "dt_next",
]


@dataclass
class Geom:
    obs: pd.DataFrame        # valid GPS rows sorted by time, projected
    arc_ids: np.ndarray      # arc ids that have valid geometry
    lines: np.ndarray        # shapely LineStrings, aligned with arc_ids
    arc_len: np.ndarray
    arc_bearing: np.ndarray
    n_arcs_total: int
    epsg: int


def project_trip(trip: Trip) -> Geom:
    tr = trip.track.copy()
    tr["obs_idx"] = np.arange(len(tr))                      # original row number in the .track file
    ok = (np.isfinite(tr[["lon", "lat", "ts"]]).all(axis=1)
          & tr.lon.between(-180, 180) & tr.lat.between(-90, 90))
    tr = tr[ok].sort_values("ts", kind="stable").reset_index(drop=True)
    if tr.empty:
        raise ValueError("no valid GPS rows")

    nd = trip.nodes
    nlon, nlat = nd.lon.to_numpy(), nd.lat.to_numpy()
    nok = np.isfinite(nlon) & np.isfinite(nlat) & (np.abs(nlon) <= 180) & (np.abs(nlat) <= 90)
    tf, epsg = make_projector(np.r_[nlon[nok], tr.lon.to_numpy()], np.r_[nlat[nok], tr.lat.to_numpy()])
    nx, ny = np.full(len(nd), np.nan), np.full(len(nd), np.nan)
    nx[nok], ny[nok] = tf.transform(nlon[nok], nlat[nok])
    x, y = tf.transform(tr.lon.to_numpy(), tr.lat.to_numpy())
    tr["x"], tr["y"] = x, y

    # GPS motion features (central difference; one-sided at the ends)
    t, n = tr.ts.to_numpy(), len(tr)
    i = np.arange(n); p = np.maximum(i - 1, 0); q = np.minimum(i + 1, n - 1)
    dx, dy, dt = x[q] - x[p], y[q] - y[p], t[q] - t[p]
    move = np.hypot(dx, dy)
    tr["speed_mps"] = np.divide(move, dt, out=np.zeros(n), where=dt > 0)
    tr["heading"] = bearing_deg(dx, dy)
    tr["heading_valid"] = move >= config.MIN_MOVE_M
    tr["dt_prev"] = np.r_[0.0, np.diff(t)]
    tr["dt_next"] = np.r_[np.diff(t), 0.0]

    a, b = trip.arcs.from_node.to_numpy(), trip.arcs.to_node.to_numpy()
    ok_arc = (a >= 0) & (a < len(nd)) & (b >= 0) & (b < len(nd))
    ok_arc[ok_arc] = nok[a[ok_arc]] & nok[b[ok_arc]]
    arc_ids = np.flatnonzero(ok_arc)
    x0, y0, x1, y1 = nx[a[arc_ids]], ny[a[arc_ids]], nx[b[arc_ids]], ny[b[arc_ids]]
    if len(arc_ids):
        lines = shapely.linestrings(np.stack([np.stack([x0, y0], 1), np.stack([x1, y1], 1)], 1))
    else:
        lines = np.array([], dtype=object)
    return Geom(tr, arc_ids, lines, np.hypot(x1 - x0, y1 - y0),
                bearing_deg(x1 - x0, y1 - y0), len(trip.arcs), epsg)


def align_route(g: Geom, route, pts):
    """Assign each GPS point a reference arc: monotone (non-decreasing) walk along the
    reference route minimising total point-to-arc distance (DP).
    Returns (ref_arc, ref_dist_m) per observation, or None if no usable route."""
    if route is None or len(route) == 0 or len(g.lines) == 0:
        return None
    lookup = np.full(g.n_arcs_total, -1, dtype=np.int64)
    lookup[g.arc_ids] = np.arange(len(g.arc_ids))
    r = route[(route >= 0) & (route < g.n_arcs_total)]
    pos = lookup[r]
    r, pos = r[pos >= 0], pos[pos >= 0]
    if len(r) == 0:
        return None
    D = shapely.distance(pts[:, None], g.lines[pos][None, :])      # n_obs x n_route_arcs
    n, m = D.shape
    ar = np.arange(m)
    cost, back = D[0].copy(), np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        cm = np.minimum.accumulate(cost)                            # best cost over arcs <= j
        back[i] = np.maximum.accumulate(np.where(cost == cm, ar, 0))  # argmin of that prefix
        cost = D[i] + cm
    j = int(np.argmin(cost))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i][j]
    return r[path], D[np.arange(n), path]


def build_trip_tables(trip: Trip, radius: float = config.RADIUS_M, max_cand: int = config.MAX_CAND):
    """Returns (obs_df, cand_df) for one trip."""
    g = project_trip(trip)
    obs = g.obs.copy()
    n = len(obs)
    pts = shapely.points(obs.x.to_numpy(), obs.y.to_numpy())

    ref = align_route(g, trip.route, pts)
    has_ref = ref is not None
    obs["ref_arc"] = ref[0] if has_ref else -1
    obs["ref_dist_m"] = ref[1] if has_ref else np.nan

    if len(g.lines):
        oi, ti = STRtree(g.lines).query(shapely.buffer(pts, radius), predicate="intersects")
        d = shapely.distance(pts[oi], g.lines[ti])
        keep = d <= radius
        oi, ti, d = oi[keep], ti[keep], d[keep]
        frac = shapely.line_locate_point(g.lines[ti], pts[oi], normalized=True)
    else:
        oi = ti = np.array([], dtype=np.int64); d = frac = np.array([])

    c = pd.DataFrame({"obs_pos": oi, "arc_pos": ti, "dist_m": d, "proj_frac": frac})
    c = c.sort_values(["obs_pos", "dist_m"], kind="stable").reset_index(drop=True)
    grp = c.groupby("obs_pos")["dist_m"]
    c["dist_rank"] = c.groupby("obs_pos").cumcount()
    c["n_cand"] = grp.transform("size")                  # density before truncation
    c["dist_minus_min"] = c.dist_m - grp.transform("min")
    c = c[c.dist_rank < max_cand].reset_index(drop=True)

    ap, op = c.arc_pos.to_numpy(), c.obs_pos.to_numpy()
    br = g.arc_bearing[ap]
    hv = obs.heading_valid.to_numpy()[op]
    hd = np.where(hv, angdiff(obs.heading.to_numpy()[op], br), 90.0)   # 90 = uninformative
    c["arc_id"] = g.arc_ids[ap]
    c["arc_len_m"] = g.arc_len[ap]
    c["dist_to_end_m"] = c.arc_len_m * (1 - c.proj_frac)
    c["is_nearest"] = (c.dist_rank == 0).astype(int)
    c["heading_diff"] = hd
    c["cos_heading_diff"] = np.cos(np.radians(hd))
    c["heading_valid"] = hv.astype(int)
    c["arc_bearing_sin"] = np.sin(np.radians(br))
    c["arc_bearing_cos"] = np.cos(np.radians(br))
    for col in ("speed_mps", "dt_prev", "dt_next"):
        c[col] = obs[col].to_numpy()[op]
    c["obs_idx"] = obs.obs_idx.to_numpy()[op]
    c["label"] = (c.arc_id.to_numpy() == obs.ref_arc.to_numpy()[op]).astype(int) if has_ref else -1

    obs["n_cand_kept"] = c.groupby("obs_pos").size().reindex(range(n), fill_value=0).to_numpy()
    if has_ref:
        hit = c[c.label == 1].groupby("obs_pos").size().reindex(range(n), fill_value=0)
        obs["ref_in_cand"] = (hit.to_numpy() > 0).astype(int)
    else:
        obs["ref_in_cand"] = -1

    obs.insert(0, "trip_id", trip.trip_id)
    c.insert(0, "trip_id", trip.trip_id)
    obs = obs[["trip_id", "obs_idx", "ts", "lon", "lat", "x", "y", "speed_mps", "heading",
               "heading_valid", "ref_arc", "ref_dist_m", "n_cand_kept", "ref_in_cand"]]
    c = c[["trip_id", "obs_idx", "arc_id"] + FEATURE_COLS + ["label"]]
    return obs, c
