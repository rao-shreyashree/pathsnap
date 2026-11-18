"""Hidden Markov Model (HMM) Route Reconstruction Decoder.

Member D — `d-rf-hmm`
Implements Newson-Krumm transition modeling, candidate emission normalization,
time-gap chain break handling, and Viterbi sequence decoding.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from src.config import (
    HMM_BETA_M,
    HMM_EMISSION_WEIGHT,
    HMM_EMISSION_EPSILON,
    HMM_MAX_VELOCITY_MPS,
    HMM_TIME_GAP_THRESHOLD_S,
    HMM_DISCONNECTED_LOG_PENALTY,
    HMM_UTURN_LOG_PENALTY,
)
from member_d.road_graph import RoadGraph
from member_d.viterbi import viterbi_decode, ViterbiResult


class HMMDecoder:
    """HMM Map-Matching Decoder combining RF emissions and topological road transitions."""

    def __init__(
        self,
        road_graph: RoadGraph,
        beta: float = HMM_BETA_M,
        emission_weight: float = HMM_EMISSION_WEIGHT,
        emission_epsilon: float = HMM_EMISSION_EPSILON,
        v_max_mps: float = HMM_MAX_VELOCITY_MPS,
        time_gap_thresh_s: float = HMM_TIME_GAP_THRESHOLD_S,
        disconnected_penalty: float = HMM_DISCONNECTED_LOG_PENALTY,
        uturn_penalty: float = HMM_UTURN_LOG_PENALTY,
    ) -> None:
        self.road_graph = road_graph
        self.beta = float(beta)
        self.emission_weight = float(emission_weight)
        self.emission_epsilon = float(emission_epsilon)
        self.v_max_mps = float(v_max_mps)
        self.time_gap_thresh_s = float(time_gap_thresh_s)
        self.disconnected_penalty = float(disconnected_penalty)
        self.uturn_penalty = float(uturn_penalty)

    def decode(
        self,
        candidates_by_obs: Sequence[Sequence[int]],
        emission_scores_by_obs: Sequence[Sequence[float]],
        track_coords_xy: np.ndarray | None = None,
        timestamps: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Decode optimal global route sequence from candidate arcs and emission scores.
        
        Parameters:
          candidates_by_obs: List of candidate arc IDs per observation
          emission_scores_by_obs: Aligned classifier prediction scores / probabilities
          track_coords_xy: Optional (N, 2) projected metric coordinates of GPS fixes
          timestamps: Optional list of timestamps in seconds
          
        Returns:
          Dictionary containing:
            'decoded_route': list[int] of predicted arc_id per observation
            'sequence_log_score': float
            'success': bool
            'n_breaks': int
            'warnings': list[str]
        """
        T = len(candidates_by_obs)
        warnings: list[str] = []

        if T == 0:
            return {
                "decoded_route": [],
                "sequence_log_score": 0.0,
                "success": True,
                "n_breaks": 0,
                "warnings": ["Empty observation sequence"],
            }

        # 1. Normalize emission scores over each observation's candidate set
        log_emissions: list[np.ndarray] = []
        cleaned_candidates: list[list[int]] = []

        for t in range(T):
            cands = list(candidates_by_obs[t])
            scores = np.array(emission_scores_by_obs[t], dtype=np.float64)

            if len(cands) == 0:
                warnings.append(f"Observation {t}: Empty candidate set; using default fallback arc.")
                # Fallback to any valid arc in graph if available
                fallback_arc = int(self.road_graph._arc_ids[0]) if self.road_graph.num_arcs > 0 else 0
                cleaned_candidates.append([fallback_arc])
                log_emissions.append(np.array([0.0]))
                continue

            # Ensure non-negative scores and prevent zero-sum division
            scores = np.maximum(scores, 0.0)
            score_sum = np.sum(scores)
            if score_sum > 0:
                probs = scores / score_sum
            else:
                probs = np.full(len(cands), 1.0 / len(cands))

            # Apply epsilon floor and compute weighted log emissions
            probs = np.maximum(probs, self.emission_epsilon)
            log_emiss = self.emission_weight * np.log(probs)

            cleaned_candidates.append(cands)
            log_emissions.append(log_emiss)

        # 2. Check for time gaps to determine chain break flags
        chain_break_flags = [False] * T
        if timestamps is not None and len(timestamps) == T:
            for t in range(1, T):
                dt = timestamps[t] - timestamps[t - 1]
                if dt > self.time_gap_thresh_s or dt < 0:
                    chain_break_flags[t] = True
                    warnings.append(f"Time gap of {dt:.1f}s detected at step {t}; broken Markov chain.")

        # 3. Define transition log-probability evaluation function
        def transition_log_prob(t: int, from_arc: int, to_arc: int) -> float:
            # Case A: Same road segment (staying on same arc)
            if from_arc == to_arc:
                return 0.0

            # Case B: Immediate topological successor
            is_immediate = self.road_graph.is_valid_transition(from_arc, to_arc)

            # Check if this transition is a reverse U-turn onto twin arc
            twin_penalty = 0.0
            if self.road_graph.reverse_twin.get(from_arc) == to_arc:
                twin_penalty = self.uturn_penalty

            # If track coordinates are provided, compute Newson-Krumm distance discrepancy
            if track_coords_xy is not None and t < len(track_coords_xy):
                p_prev = track_coords_xy[t - 1]
                p_curr = track_coords_xy[t]
                d_gc = float(np.sqrt((p_curr[0] - p_prev[0]) ** 2 + (p_curr[1] - p_prev[1]) ** 2))

                # Maximum feasible distance vehicle could travel in dt
                dt = (timestamps[t] - timestamps[t - 1]) if (timestamps is not None and t < len(timestamps)) else 1.0
                dt = max(dt, 0.1)
                max_reach = min(1500.0, self.v_max_mps * dt + 50.0)

                d_net = self.road_graph.shortest_network_distance(from_arc, to_arc, max_dist_m=max_reach)

                if d_net is None:
                    # Topologically unreachable within velocity constraint
                    return self.disconnected_penalty + twin_penalty

                # Distance discrepancy: |d_net - d_gc|
                delta_d = abs(d_net - d_gc)
                log_prob = -(delta_d / self.beta) + twin_penalty
                return max(log_prob, self.disconnected_penalty)
            else:
                # Topological connectivity only
                if is_immediate:
                    return 0.0 + twin_penalty
                
                # Check 2-3 hop reachability
                d_net = self.road_graph.shortest_network_distance(from_arc, to_arc, max_dist_m=300.0)
                if d_net is not None:
                    return -(d_net / self.beta) + twin_penalty
                return self.disconnected_penalty + twin_penalty

        # 4. Run Viterbi dynamic programming
        viterbi_res = viterbi_decode(
            candidate_states=cleaned_candidates,
            log_emissions=log_emissions,
            transition_log_prob_fn=transition_log_prob,
            chain_break_flags=chain_break_flags,
            disconnected_penalty=self.disconnected_penalty,
        )

        return {
            "decoded_route": [int(x) for x in viterbi_res.best_sequence],
            "sequence_log_score": round(viterbi_res.best_log_score, 4),
            "success": True,
            "n_breaks": len(viterbi_res.chain_breaks),
            "chain_breaks": viterbi_res.chain_breaks,
            "warnings": warnings,
        }
