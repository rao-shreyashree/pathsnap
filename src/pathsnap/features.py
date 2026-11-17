"""Feature extraction for candidate road arcs.

Transforms (GPS observation, candidate arc) pairs into invariant numerical feature vectors.
Contains ZERO ground truth access or label dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pathsnap.geo import (
    compute_track_bearings,
    point_to_segment_distance,
)
from src.pathsnap.network import SpatialNetworkIndex
from src.pathsnap.candidates import CandidateSet

FEATURE_NAMES = [
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


def extract_features_for_trip(
    track_df: pd.DataFrame,
    px: np.ndarray,
    py: np.ndarray,
    candidates: CandidateSet,
    spatial_index: SpatialNetworkIndex,
    has_reverse_twin: np.ndarray,
    in_degrees: np.ndarray,
    out_degrees: np.ndarray,
    cropped_arcs_df: pd.DataFrame,
) -> np.ndarray:
    """Compute (N_cand, 20) feature matrix for all candidate rows of a trip."""
    n_cands = candidates.total_candidates
    n_obs = len(px)
    if n_cands == 0 or n_obs == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    # 1. Track kinematics (speeds, headings, dt)
    t_vals = track_df["t"].to_numpy()
    dts_fwd = np.zeros(n_obs, dtype=np.float64)
    dts_bwd = np.zeros(n_obs, dtype=np.float64)
    speeds = np.zeros(n_obs, dtype=np.float64)

    if n_obs > 1:
        dt_raw = np.diff(t_vals)
        dt_safe = np.maximum(dt_raw, 0.001)

        dx = np.diff(px)
        dy = np.diff(py)
        dist_steps = np.sqrt(dx * dx + dy * dy)
        step_speeds = dist_steps / dt_safe

        dts_fwd[:-1] = dt_raw
        dts_fwd[-1] = dt_raw[-1]
        dts_bwd[1:] = dt_raw
        dts_bwd[0] = dt_raw[0]

        speeds[:-1] = step_speeds
        speeds[-1] = step_speeds[-1]

    track_bearings = compute_track_bearings(px, py, speeds)

    # 2. Extract per-candidate attributes
    obs_idx = candidates.obs_indices
    arc_idx = candidates.arc_indices
    perp_dist = candidates.perp_dists
    along_t = candidates.along_arc_ts

    arc_x1 = spatial_index.arc_x1[arc_idx]
    arc_y1 = spatial_index.arc_y1[arc_idx]
    arc_x2 = spatial_index.arc_x2[arc_idx]
    arc_y2 = spatial_index.arc_y2[arc_idx]
    arc_len = spatial_index.arc_lengths[arc_idx]
    arc_bearing = spatial_index.arc_bearings[arc_idx]

    # Node degrees
    from_col = "crop_from_node" if "crop_from_node" in cropped_arcs_df.columns else "from_node"
    to_col = "crop_to_node" if "crop_to_node" in cropped_arcs_df.columns else "to_node"
    u_nodes = cropped_arcs_df[from_col].to_numpy()[arc_idx]
    v_nodes = cropped_arcs_df[to_col].to_numpy()[arc_idx]

    from_deg = out_degrees[u_nodes]
    to_deg = in_degrees[v_nodes]
    twin_flag = has_reverse_twin[arc_idx].astype(np.float32)

    # Distances to endpoints
    dist_to_start = along_t * arc_len
    dist_to_end = (1.0 - along_t) * arc_len

    # Heading alignment
    obs_bearing = track_bearings[obs_idx]
    angle_diff = np.abs(obs_bearing - arc_bearing) % 360.0
    angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)  # in [0, 180]
    rad_diff = np.radians(angle_diff)
    cos_heading = np.cos(rad_diff)
    sin_heading = np.sin(rad_diff)

    # Per-observation relative metrics
    dist_rank = np.zeros(n_cands, dtype=np.float32)
    dist_minus_min = np.zeros(n_cands, dtype=np.float32)
    rel_dist = np.zeros(n_cands, dtype=np.float32)
    num_cands = np.zeros(n_cands, dtype=np.float32)
    context_is_nearest = np.zeros(n_cands, dtype=np.float32)
    context_mean_dist = np.zeros(n_cands, dtype=np.float32)

    # Track nearest arc per observation for context feature
    nearest_arc_idx_per_obs = np.zeros(n_obs, dtype=np.int32)
    offsets = candidates.cand_offsets

    for i in range(n_obs):
        st = offsets[i]
        en = offsets[i + 1]
        k = en - st
        if k > 0:
            num_cands[st:en] = k
            dist_rank[st:en] = np.arange(k, dtype=np.float32)
            min_d = perp_dist[st]
            dist_minus_min[st:en] = perp_dist[st:en] - min_d
            rel_dist[st:en] = perp_dist[st:en] / (min_d + 1.0)
            nearest_arc_idx_per_obs[i] = arc_idx[st]

    # Context features across 3-step window [i-1, i, i+1]
    for i in range(n_obs):
        st = offsets[i]
        en = offsets[i + 1]
        if en <= st:
            continue

        prev_i = max(0, i - 1)
        next_i = min(n_obs - 1, i + 1)
        nbr_obs = [prev_i, i, next_i]

        cur_arc_idxs = arc_idx[st:en]
        c_x1 = arc_x1[st:en]
        c_y1 = arc_y1[st:en]
        c_x2 = arc_x2[st:en]
        c_y2 = arc_y2[st:en]

        # Check whether this arc is nearest for neighbors
        is_near_count = np.zeros(len(cur_arc_idxs), dtype=np.float32)
        dist_sum = np.zeros(len(cur_arc_idxs), dtype=np.float32)

        for n_i in nbr_obs:
            is_near_count += (cur_arc_idxs == nearest_arc_idx_per_obs[n_i]).astype(np.float32)
            d_nbr, _, _, _ = point_to_segment_distance(px[n_i], py[n_i], c_x1, c_y1, c_x2, c_y2)
            dist_sum += d_nbr

        context_is_nearest[st:en] = is_near_count / len(nbr_obs)
        context_mean_dist[st:en] = dist_sum / len(nbr_obs)

    # Assemble feature matrix
    X = np.column_stack([
        perp_dist.astype(np.float32),
        dist_rank.astype(np.float32),
        dist_minus_min.astype(np.float32),
        rel_dist.astype(np.float32),
        along_t.astype(np.float32),
        dist_to_start.astype(np.float32),
        dist_to_end.astype(np.float32),
        arc_len.astype(np.float32),
        speeds[obs_idx].astype(np.float32),
        dts_bwd[obs_idx].astype(np.float32),
        dts_fwd[obs_idx].astype(np.float32),
        angle_diff.astype(np.float32),
        cos_heading.astype(np.float32),
        sin_heading.astype(np.float32),
        num_cands.astype(np.float32),
        twin_flag.astype(np.float32),
        from_deg.astype(np.float32),
        to_deg.astype(np.float32),
        context_mean_dist.astype(np.float32),
        context_is_nearest.astype(np.float32),
    ])

    return X
