"""Unit tests for RoadGraph (Phase A).

Tests toy graphs with known topology and verified real trip samples.
"""
from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from member_d.road_graph import RoadGraph, project_equirectangular, point_to_segment_distance
from member_d.config import DATA_RAW_TRAIN


@pytest.fixture
def toy_graph() -> RoadGraph:
    """Create a hand-crafted 5-node, 6-arc directed graph for deterministic tests.
    
    Topology:
      Node 0: (lon=0.0, lat=0.0)
      Node 1: (lon=0.001, lat=0.0)
      Node 2: (lon=0.001, lat=0.001)
      Node 3: (lon=0.0, lat=0.001)
      Node 4: (lon=0.005, lat=0.005)  [Isolated / disconnected component]

    Arcs:
      Arc 0: 0 -> 1 (forward)
      Arc 1: 1 -> 0 (reverse twin of Arc 0)
      Arc 2: 1 -> 2 (connected to Arc 0)
      Arc 3: 2 -> 3 (connected to Arc 2)
      Arc 4: 3 -> 0 (forms cycle back to 0)
      Arc 5: 4 -> 4 (self-loop on isolated node 4)
    """
    nodes_df = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4],
        "lon": [0.0, 0.001, 0.001, 0.0, 0.005],
        "lat": [0.0, 0.0, 0.001, 0.001, 0.005],
    })
    arcs_df = pd.DataFrame({
        "arc_id": [0, 1, 2, 3, 4, 5],
        "from_node": [0, 1, 1, 2, 3, 4],
        "to_node":   [1, 0, 2, 3, 0, 4],
    })
    return RoadGraph(nodes_df, arcs_df, lon0=0.0, lat0=0.0)


def test_toy_graph_attributes(toy_graph: RoadGraph):
    assert toy_graph.num_nodes == 5
    assert toy_graph.num_arcs == 6
    assert toy_graph.is_valid_arc(0)
    assert toy_graph.is_valid_arc(5)
    assert not toy_graph.is_valid_arc(999)


def test_successors_and_predecessors(toy_graph: RoadGraph):
    # Arc 0 goes 0 -> 1. Nodes out from 1 are Arc 1 (1->0) and Arc 2 (1->2)
    succs = toy_graph.successors[0]
    assert set(succs) == {1, 2}

    # Arc 2 goes 1 -> 2. Nodes out from 2 is Arc 3 (2->3)
    assert toy_graph.successors[2] == [3]

    # Predecessors of Arc 3 (2->3) must include Arc 2
    preds = toy_graph.predecessors[3]
    assert 2 in preds


def test_reverse_twin_detection(toy_graph: RoadGraph):
    # Arc 0 (0->1) has twin Arc 1 (1->0)
    assert toy_graph.reverse_twin[0] == 1
    assert toy_graph.reverse_twin[1] == 0
    # Arc 2 (1->2) has no reverse twin
    assert toy_graph.reverse_twin[2] is None


def test_valid_and_invalid_transitions(toy_graph: RoadGraph):
    # Arc 0 -> Arc 2 is valid (0->1, 1->2)
    assert toy_graph.is_valid_transition(0, 2)
    # Arc 0 -> Arc 3 is invalid (needs intermediate Arc 2)
    assert not toy_graph.is_valid_transition(0, 3)
    # Invalid arc ID returns False
    assert not toy_graph.is_valid_transition(0, 999)
    assert not toy_graph.is_valid_transition(-1, 2)


def test_shortest_network_distance(toy_graph: RoadGraph):
    # Same arc = 0.0
    assert toy_graph.shortest_network_distance(0, 0) == 0.0
    # Immediate successor = 0.0
    assert toy_graph.shortest_network_distance(0, 2) == 0.0
    # Two hops: Arc 0 (0->1) -> Arc 2 (1->2) -> Arc 3 (2->3)
    # Distance to reach Arc 3's start is Arc 2's length
    geom2 = toy_graph.get_arc_geometry(2)
    d_0_to_3 = toy_graph.shortest_network_distance(0, 3)
    assert d_0_to_3 is not None
    assert pytest.approx(d_0_to_3, rel=1e-3) == geom2["length_m"]
    # Disconnected transition to Arc 5 (on isolated component)
    assert toy_graph.shortest_network_distance(0, 5) is None


def test_integrity_audit(toy_graph: RoadGraph):
    audit = toy_graph.validate_integrity()
    assert audit["num_nodes"] == 5
    assert audit["num_arcs"] == 6
    assert audit["invalid_from_nodes"] == 0
    assert audit["invalid_to_nodes"] == 0
    assert audit["self_loops"] == 1  # Arc 5 is 4->4


def test_real_trip_graph():
    """Verify RoadGraph loading on an actual permitted training trip."""
    trip_dir = DATA_RAW_TRAIN / "00000001"
    if not trip_dir.exists():
        pytest.skip("Training data directory not found")

    graph = RoadGraph.from_trip_directory(trip_dir, crop=True)
    assert graph.num_nodes > 0
    assert graph.num_arcs > 0

    audit = graph.validate_integrity()
    assert audit["invalid_from_nodes"] == 0
    assert audit["invalid_to_nodes"] == 0
    assert audit["reverse_twin_percentage"] > 50.0  # Most road segments have twins
