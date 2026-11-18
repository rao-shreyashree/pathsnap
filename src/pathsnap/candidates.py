"""Candidate road arc generation for GPS observations.

For each GPS fix:
1. Spatially queries the KDTree index within candidate search radius R.
2. If fewer than min_cand found, adaptively widens the radius up to R_max.
3. Computes exact perpendicular distance and projection point onto the candidate segment.
4. Ranks candidates and selects the top K nearest candidates.

Zero ground truth access: candidate generation is strictly spatial.
"""
from __future__ import annotations

import numpy as np
from src.pathsnap.geo import point_to_segment_distance
from src.pathsnap.network import SpatialNetworkIndex


class CandidateSet:
    """Holds candidate arcs and geometric projections for all observations in a trip."""

    def __init__(
        self,
        obs_indices: np.ndarray,
        arc_ids: np.ndarray,
        arc_indices: np.ndarray,
        perp_dists: np.ndarray,
        along_arc_ts: np.ndarray,
        proj_xs: np.ndarray,
        proj_ys: np.ndarray,
        cand_offsets: np.ndarray,  # 1D array of length N_obs + 1 giving slice ranges per obs
    ):
        self.obs_indices = obs_indices
        self.arc_ids = arc_ids
        self.arc_indices = arc_indices
        self.perp_dists = perp_dists
        self.along_arc_ts = along_arc_ts
        self.proj_xs = proj_xs
        self.proj_ys = proj_ys
        self.cand_offsets = cand_offsets

    @property
    def total_candidates(self) -> int:
        return len(self.obs_indices)

    def get_candidates_for_obs(self, obs_idx: int) -> dict[str, np.ndarray]:
        """Retrieve candidate records for a specific observation index."""
        start = self.cand_offsets[obs_idx]
        end = self.cand_offsets[obs_idx + 1]
        return {
            "arc_id": self.arc_ids[start:end],
            "arc_idx": self.arc_indices[start:end],
            "perp_dist": self.perp_dists[start:end],
            "along_arc_t": self.along_arc_ts[start:end],
            "proj_x": self.proj_xs[start:end],
            "proj_y": self.proj_ys[start:end],
        }


def generate_candidates(
    px: np.ndarray,
    py: np.ndarray,
    spatial_index: SpatialNetworkIndex,
    radius_m: float = 50.0,
    max_k: int = 10,
    min_cand: int = 3,
    max_radius_m: float = 250.0,
) -> CandidateSet:
    """Generate up to max_k nearest candidate road segments for each GPS fix.
    
    Parameters:
      px, py: projected GPS coordinates in metres (1D arrays of length N)
      spatial_index: precomputed SpatialNetworkIndex
      radius_m: primary search radius in metres
      max_k: maximum number of candidate arcs to keep per observation
      min_cand: minimum number of candidates required per observation
      max_radius_m: ceiling for adaptive radius widening
    """
    n_obs = len(px)
    tree = spatial_index.tree
    sample_to_arc = spatial_index.sample_to_arc_idx

    obs_idx_list: list[np.ndarray] = []
    arc_id_list: list[np.ndarray] = []
    arc_idx_list: list[np.ndarray] = []
    perp_dist_list: list[np.ndarray] = []
    along_t_list: list[np.ndarray] = []
    proj_x_list: list[np.ndarray] = []
    proj_y_list: list[np.ndarray] = []
    cand_offsets = np.zeros(n_obs + 1, dtype=np.int32)

    total_cands = 0

    for i in range(n_obs):
        cur_px = px[i]
        cur_py = py[i]

        cur_r = radius_m
        unique_arc_indices: np.ndarray = np.array([], dtype=np.int32)

        # Query KDTree with adaptive widening if needed
        while True:
            pt_indices = tree.query_ball_point([cur_px, cur_py], r=cur_r)
            if pt_indices:
                candidate_arc_idxs = np.unique(sample_to_arc[pt_indices])
            else:
                candidate_arc_idxs = np.array([], dtype=np.int32)

            if len(candidate_arc_idxs) >= min_cand or cur_r >= max_radius_m:
                unique_arc_indices = candidate_arc_idxs
                break
            cur_r = min(cur_r * 1.5, max_radius_m)

        # Fallback if still empty: query nearest single point
        if len(unique_arc_indices) == 0:
            _, nearest_sample = tree.query([cur_px, cur_py], k=1)
            unique_arc_indices = np.array([sample_to_arc[nearest_sample]], dtype=np.int32)

        # Compute exact perpendicular distance to all candidate arc segments
        c_x1 = spatial_index.arc_x1[unique_arc_indices]
        c_y1 = spatial_index.arc_y1[unique_arc_indices]
        c_x2 = spatial_index.arc_x2[unique_arc_indices]
        c_y2 = spatial_index.arc_y2[unique_arc_indices]

        dists, ts, qxs, qys = point_to_segment_distance(
            cur_px, cur_py, c_x1, c_y1, c_x2, c_y2
        )

        # Sort by distance and truncate to max_k
        sort_order = np.argsort(dists)[:max_k]

        k_kept = len(sort_order)
        cand_arc_idxs = unique_arc_indices[sort_order]
        cand_arc_ids = spatial_index.arc_ids[cand_arc_idxs]
        cand_dists = dists[sort_order]
        cand_ts = ts[sort_order]
        cand_qxs = qxs[sort_order]
        cand_qys = qys[sort_order]

        obs_idx_list.append(np.full(k_kept, i, dtype=np.int32))
        arc_id_list.append(cand_arc_ids)
        arc_idx_list.append(cand_arc_idxs)
        perp_dist_list.append(cand_dists)
        along_t_list.append(cand_ts)
        proj_x_list.append(cand_qxs)
        proj_y_list.append(cand_qys)

        total_cands += k_kept
        cand_offsets[i + 1] = total_cands

    return CandidateSet(
        obs_indices=np.concatenate(obs_idx_list) if obs_idx_list else np.array([], dtype=np.int32),
        arc_ids=np.concatenate(arc_id_list) if arc_id_list else np.array([], dtype=np.int32),
        arc_indices=np.concatenate(arc_idx_list) if arc_idx_list else np.array([], dtype=np.int32),
        perp_dists=np.concatenate(perp_dist_list) if perp_dist_list else np.array([], dtype=np.float64),
        along_arc_ts=np.concatenate(along_t_list) if along_t_list else np.array([], dtype=np.float64),
        proj_xs=np.concatenate(proj_x_list) if proj_x_list else np.array([], dtype=np.float64),
        proj_ys=np.concatenate(proj_y_list) if proj_y_list else np.array([], dtype=np.float64),
        cand_offsets=cand_offsets,
    )
