"""Unit tests for FastAPI Integration Interface (Phase G)."""
from __future__ import annotations

import pytest
import pandas as pd
from member_d.road_graph import RoadGraph
from member_d.integration import decode_route


@pytest.fixture
def toy_graph() -> RoadGraph:
    nodes_df = pd.DataFrame({"node_id": [0, 1], "lon": [0.0, 0.001], "lat": [0.0, 0.0]})
    arcs_df = pd.DataFrame({"arc_id": [0], "from_node": [0], "to_node": [1]})
    return RoadGraph(nodes_df, arcs_df, lon0=0.0, lat0=0.0)


def test_decode_route_success(toy_graph: RoadGraph):
    res = decode_route(
        candidates=[[0]],
        emission_scores=[[1.0]],
        road_graph=toy_graph,
        timestamps=[1.0],
    )
    assert res["success"] is True
    assert res["decoded_route"] == [0]
    assert isinstance(res["sequence_log_score"], float)


def test_decode_route_mismatched_lengths(toy_graph: RoadGraph):
    res = decode_route(
        candidates=[[0], [0]],
        emission_scores=[[1.0]],
        road_graph=toy_graph,
    )
    assert res["success"] is False
    assert res["decoded_route"] == []
    assert len(res["warnings"]) > 0


def test_decode_route_invalid_graph_type():
    with pytest.raises(TypeError):
        decode_route([[0]], [[1.0]], road_graph="not_a_graph")  # type: ignore


def test_decode_route_empty(toy_graph: RoadGraph):
    res = decode_route(candidates=[], emission_scores=[], road_graph=toy_graph)
    assert res["success"] is True
    assert res["decoded_route"] == []
