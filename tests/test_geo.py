import numpy as np
from backend.geo import arcs_to_geojson, continuity, pointwise_score

NODES = np.array([[10.0, 50.0], [10.1, 50.0], [10.2, 50.0], [11.0, 51.0], [11.1, 51.0]])
ARCS = np.array([[0, 1], [1, 2], [3, 4]])


def test_linestring_continuous_and_dedup():
    g = arcs_to_geojson([0, 0, 1], NODES, ARCS)
    assert g == {"type": "LineString", "coordinates": [[10.0, 50.0], [10.1, 50.0], [10.2, 50.0]]}


def test_multilinestring_on_gap():
    g = arcs_to_geojson([0, 1, 2], NODES, ARCS)
    assert g["type"] == "MultiLineString" and len(g["coordinates"]) == 2


def test_pointwise_and_missing():
    ref = {0: 5, 1: 6, 2: -1, 3: float("nan"), 4: 7}
    assert pointwise_score({0: 5, 1: 9}, ref) == (1, 3)     # obs 4 has no prediction -> wrong


def test_continuity():
    assert continuity({0: 0, 1: 1, 2: 2}, ARCS) == 0.5


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("ok")


def test_segments_lat_lon_order():
    from backend.geo import arcs_to_segments
    assert arcs_to_segments([0, 0, 1], NODES, ARCS) == [
        [[50.0, 10.0], [50.0, 10.1]], [[50.0, 10.1], [50.0, 10.2]]]
