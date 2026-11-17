"""Ground-truth reference label handling and monotone route alignment.

P3 RULE: .route files, reference arcs, and per-observation labels may be read
ONLY in this module (src/pathsnap/labels.py) and in src/pathsnap/scoring.py.
They must NEVER reach feature code, models at inference, HMM, API, or UI.

Implements:
- load_route(file_path): loads the reference arc sequence.
- derive_labels_monotone(track_df, nodes_df, arcs_df, route_arcs):
    Monotone dynamic programming alignment of GPS observations to the route sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd

from src.pathsnap.geo import (
    get_centroid,
    project_equirectangular,
    point_to_segment_distance,
)


class AlignmentResult(NamedTuple):
    ref_arc_ids: np.ndarray        # length N_obs: matched ground-truth arc_id per observation
    ref_route_indices: np.ndarray  # length N_obs: index in route array
    distances: np.ndarray          # length N_obs: perpendicular distance to assigned route arc
    mean_dist_m: float
    p95_dist_m: float
    share_gt_30m: float
    share_uncertain: float         # share of points where distance > 50m


def load_route(route_path: str | Path) -> np.ndarray:
    """Load a .route reference file: integer sequence of traversed arc_ids."""
    p = Path(route_path)
    if not p.exists():
        raise FileNotFoundError(f"Route file not found: {p}")
    
    route = np.loadtxt(p, dtype=np.int32)
    if route.ndim == 0:
        route = np.array([route], dtype=np.int32)
    return route


def derive_labels_monotone(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    route_arcs: np.ndarray,
    v_max_mps: float = 45.0,  # ~160 km/h
    margin_m: float = 120.0,
    cost_cap_m: float = 120.0,
) -> AlignmentResult:
    """Derive pointwise reference labels via monotone dynamic programming alignment along route index.
    
    The vehicle traverses the sequence of arcs in .route in chronological order.
    Observations i = 0..N-1 are aligned to route indices j = 0..M-1 such that:
      j_{i+1} >= j_i (monotone progress along the route).
    """
    n_obs = len(track_df)
    n_route = len(route_arcs)

    if n_route == 0 or n_obs == 0:
        raise ValueError("Cannot align empty track or route")

    # Project coordinates to local metric plane
    lon0, lat0 = get_centroid(track_df["lon"].to_numpy(), track_df["lat"].to_numpy())
    px, py = project_equirectangular(track_df["lon"].to_numpy(), track_df["lat"].to_numpy(), lon0, lat0)
    t_vals = track_df["t"].to_numpy()

    node_lons = nodes_df["lon"].to_numpy()
    node_lats = nodes_df["lat"].to_numpy()
    nx, ny = project_equirectangular(node_lons, node_lats, lon0, lat0)

    # Route segment geometry
    from_nodes = arcs_df["from_node"].to_numpy()
    to_nodes = arcs_df["to_node"].to_numpy()

    r_u = from_nodes[route_arcs]
    r_v = to_nodes[route_arcs]

    rx1 = nx[r_u]
    ry1 = ny[r_u]
    rx2 = nx[r_v]
    ry2 = ny[r_v]

    r_dx = rx2 - rx1
    r_dy = ry2 - ry1
    seg_lens = np.sqrt(r_dx * r_dx + r_dy * r_dy)
    cum_dist = np.zeros(n_route + 1, dtype=np.float64)
    cum_dist[1:] = np.cumsum(seg_lens)

    # Forward DP: cost matrix (N_obs, N_route)
    # To optimize memory for long trips, we maintain running DP state
    # dp[j] = minimum alignment cost reaching route index j at observation i
    INF = 1e9
    cap_sq = cost_cap_m ** 2

    # Initialise observation 0
    dists_0, _, _, _ = point_to_segment_distance(px[0], py[0], rx1, ry1, rx2, ry2)
    # We allow starting at early route indices (e.g. first 20 or wherever distance is small)
    dp_cost = np.minimum(dists_0 ** 2, cap_sq)
    # Discourage starting unrealistically deep into the route
    start_penalty = np.linspace(0.0, 5000.0, n_route)
    dp_cost += start_penalty

    # Backpointers: backptr[i, j] = previous route index k
    backptr = np.zeros((n_obs, n_route), dtype=np.int32)

    # Precompute distances in chunks to save memory
    chunk_size = 500
    for chunk_start in range(1, n_obs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_obs)
        c_px = px[chunk_start:chunk_end, None]
        c_py = py[chunk_start:chunk_end, None]
        chunk_dists, _, _, _ = point_to_segment_distance(
            c_px, c_py, rx1[None, :], ry1[None, :], rx2[None, :], ry2[None, :]
        )
        chunk_costs = np.minimum(chunk_dists ** 2, cap_sq)

        for step in range(chunk_end - chunk_start):
            i = chunk_start + step
            dt = t_vals[i] - t_vals[i - 1]
            max_forward_dist = (v_max_mps * dt + margin_m) if dt <= 60.0 else INF

            curr_obs_cost = chunk_costs[step]
            next_dp = np.full(n_route, INF, dtype=np.float64)

            # Vectorized running minimum with reachability
            # Since j' >= j, we scan j from 0 to n_route - 1
            min_prev_cost = INF
            best_k = 0
            k_start = 0

            for j in range(n_route):
                # Expand valid k range
                while k_start <= j and (cum_dist[j] - cum_dist[k_start] > max_forward_dist):
                    k_start += 1

                # Find best k in [k_start, j]
                sub_slice = dp_cost[k_start : j + 1]
                if len(sub_slice) > 0:
                    local_min_idx = int(np.argmin(sub_slice))
                    local_min_val = sub_slice[local_min_idx]
                    k_opt = k_start + local_min_idx
                else:
                    k_opt = j
                    local_min_val = dp_cost[j]

                next_dp[j] = local_min_val + curr_obs_cost[j]
                backptr[i, j] = k_opt

            dp_cost = next_dp

    # Backtrack from final step
    best_last_j = int(np.argmin(dp_cost))
    aligned_j = np.zeros(n_obs, dtype=np.int32)
    aligned_j[-1] = best_last_j

    for i in range(n_obs - 1, 0, -1):
        aligned_j[i - 1] = backptr[i, aligned_j[i]]

    aligned_arc_ids = route_arcs[aligned_j]

    # Calculate pointwise distances to the assigned arc
    assigned_x1 = rx1[aligned_j]
    assigned_y1 = ry1[aligned_j]
    assigned_x2 = rx2[aligned_j]
    assigned_y2 = ry2[aligned_j]
    point_dists, _, _, _ = point_to_segment_distance(
        px, py, assigned_x1, assigned_y1, assigned_x2, assigned_y2
    )

    mean_dist = float(np.mean(point_dists))
    p95_dist = float(np.percentile(point_dists, 95))
    share_gt_30 = float(np.mean(point_dists > 30.0))
    share_unc = float(np.mean(point_dists > 50.0))

    return AlignmentResult(
        ref_arc_ids=aligned_arc_ids,
        ref_route_indices=aligned_j,
        distances=point_dists,
        mean_dist_m=mean_dist,
        p95_dist_m=p95_dist,
        share_gt_30m=share_gt_30,
        share_uncertain=share_unc,
    )
