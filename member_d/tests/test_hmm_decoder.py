"""Unit tests for HMM Decoder (Phase C)."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from member_d.road_graph import RoadGraph
from member_d.hmm_decoder import HMMDecoder


@pytest.fixture
def toy_graph() -> RoadGraph:
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "lon": [0.0, 0.001, 0.002, 0.003],
        "lat": [0.0, 0.0, 0.0, 0.0],
    })
    # Arcs: 0: 0->1, 1: 1->0 (twin), 2: 1->2, 3: 2->3
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1, 2, 3],
        "from_node": [0, 1, 1, 2],
        "to_node":   [1, 0, 2, 3],
    })
    return RoadGraph(nodes_df, arcs_df, lon0=0.0, lat0=0.0)


def test_hmm_decodes_connected_path(toy_graph: RoadGraph):
    decoder = HMMDecoder(road_graph=toy_graph)

    # Step 0: candidates [0, 1]. Arc 0 is forward.
    # Step 1: candidates [1, 2]. Arc 2 is successor of Arc 0.
    candidates = [[0, 1], [1, 2]]
    scores = [[0.6, 0.4], [0.4, 0.6]]

    res = decoder.decode(candidates, scores)
    assert res["success"] is True
    assert res["decoded_route"] == [0, 2]
    assert res["n_breaks"] == 0


def test_hmm_handles_large_time_gap(toy_graph: RoadGraph):
    decoder = HMMDecoder(road_graph=toy_graph, time_gap_thresh_s=60.0)

    # 100-second gap between step 0 and step 1
    candidates = [[0, 1], [1, 2]]
    scores = [[0.9, 0.1], [0.9, 0.1]]  # naive top is [0, 1], which is a U-turn
    timestamps = [10.0, 110.0]

    res = decoder.decode(candidates, scores, timestamps=timestamps)
    assert res["success"] is True
    assert res["n_breaks"] == 1
    # When chain is broken, naive top [0, 1] is chosen because transition penalty is skipped
    assert res["decoded_route"] == [0, 1]


def test_hmm_handles_empty_candidates(toy_graph: RoadGraph):
    decoder = HMMDecoder(road_graph=toy_graph)

    candidates = [[0, 1], [], [2, 3]]
    scores = [[0.8, 0.2], [], [0.7, 0.3]]

    res = decoder.decode(candidates, scores)
    assert res["success"] is True
    assert len(res["decoded_route"]) == 3
    assert len(res["warnings"]) > 0
