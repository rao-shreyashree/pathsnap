"""FastAPI-Independent Integration Interface for Member C.

Member D — `d-rf-hmm`

This module provides the exact standalone callable function `decode_route(...)`
requested by Member C for the FastAPI `/predict` backend endpoint.

Import Statement:
    from member_d.integration import decode_route

Example Usage:
    from member_d.integration import decode_route
    from member_d.road_graph import RoadGraph

    graph = RoadGraph.from_trip_directory("data/raw/student_test/00000000")
    result = decode_route(
        candidates=[[101, 102], [102, 105]],
        emission_scores=[[0.85, 0.15], [0.10, 0.90]],
        road_graph=graph,
        timestamps=[10.0, 11.0],
    )
    print("Decoded route:", result["decoded_route"])
"""
from __future__ import annotations

from typing import Any, Sequence
from member_d.road_graph import RoadGraph
from member_d.hmm_decoder import HMMDecoder


def decode_route(
    candidates: Sequence[Sequence[int]],
    emission_scores: Sequence[Sequence[float]],
    road_graph: RoadGraph,
    timestamps: Sequence[float] | None = None,
    track_coords_xy: Any | None = None,
) -> dict[str, Any]:
    """Decoupled map-matching sequence decoder for FastAPI and batch inference.
    
    Parameters:
      candidates: Candidate road arc IDs for each GPS observation (outer len T).
      emission_scores: Classifier prediction probabilities aligned with candidate IDs.
      road_graph: Constructed RoadGraph object with topological adjacency.
      timestamps: Optional sequence of timestamps in seconds (len T).
      track_coords_xy: Optional (T, 2) projected metric coordinates in metres.
      
    Returns:
      dict with keys:
        - "decoded_route": list[int] (one predicted arc ID per observation)
        - "sequence_log_score": float (total Viterbi log-likelihood)
        - "success": bool (True if decoding succeeded without fatal error)
        - "n_breaks": int (number of Markov chain breaks due to gaps/disconnections)
        - "warnings": list[str] (human-readable diagnostic messages)
    """
    # 1. Input parameter validation
    if not isinstance(road_graph, RoadGraph):
        raise TypeError(f"road_graph must be an instance of RoadGraph, got {type(road_graph).__name__}")

    n_cands = len(candidates)
    n_scores = len(emission_scores)

    if n_cands != n_scores:
        return {
            "decoded_route": [],
            "sequence_log_score": 0.0,
            "success": False,
            "n_breaks": 0,
            "warnings": [
                f"Length mismatch: candidates has {n_cands} steps, emission_scores has {n_scores} steps"
            ],
        }

    if n_cands == 0:
        return {
            "decoded_route": [],
            "sequence_log_score": 0.0,
            "success": True,
            "n_breaks": 0,
            "warnings": ["Received empty candidates array"],
        }

    # 2. Delegate to HMMDecoder
    decoder = HMMDecoder(road_graph=road_graph)
    result = decoder.decode(
        candidates_by_obs=candidates,
        emission_scores_by_obs=emission_scores,
        track_coords_xy=track_coords_xy,
        timestamps=timestamps,
    )

    return result
