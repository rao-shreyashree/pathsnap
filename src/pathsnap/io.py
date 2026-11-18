"""Robust I/O loaders for PathSnap trip data.

Files per trip:
- <trip_id>.track: whitespace/tab-separated (lon, lat, t), no headers, nominally 1 Hz.
- <trip_id>.nodes: whitespace/tab-separated (lon, lat), no headers, row index = node_id (0-based).
- <trip_id>.arcs:  whitespace/tab-separated (from_node, to_node), no headers, row index = arc_id (0-based).

Note: Ground truth reference sequences are strictly handled in labels.py and scoring.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd


class TripData(NamedTuple):
    trip_id: str
    track: pd.DataFrame  # columns: ['lon', 'lat', 't']
    nodes: pd.DataFrame  # columns: ['node_id', 'lon', 'lat']
    arcs: pd.DataFrame   # columns: ['arc_id', 'from_node', 'to_node']


def load_track(file_path: str | Path) -> pd.DataFrame:
    """Load a .track file: columns (lon, lat, t)."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Track file not found: {p}")
    
    df = pd.read_csv(
        p,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        dtype={"lon": np.float64, "lat": np.float64, "t": np.float64},
        engine="c",
    )
    return df


def load_nodes(file_path: str | Path) -> pd.DataFrame:
    """Load a .nodes file: columns (lon, lat); row index is node_id."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Nodes file not found: {p}")
    
    df = pd.read_csv(
        p,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
        engine="c",
    )
    df["node_id"] = np.arange(len(df), dtype=np.int32)
    return df[["node_id", "lon", "lat"]]


def load_arcs(file_path: str | Path) -> pd.DataFrame:
    """Load an .arcs file: columns (from_node, to_node); row index is arc_id."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Arcs file not found: {p}")
    
    df = pd.read_csv(
        p,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int32, "to_node": np.int32},
        engine="c",
    )
    df["arc_id"] = np.arange(len(df), dtype=np.int32)
    return df[["arc_id", "from_node", "to_node"]]


def load_trip(trip_dir: str | Path, trip_id: str | None = None) -> TripData:
    """Load public files (.track, .nodes, .arcs) for a trip directory."""
    d = Path(trip_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"Trip directory not found: {d}")
    
    tid = trip_id or d.name
    track_path = d / f"{tid}.track"
    nodes_path = d / f"{tid}.nodes"
    arcs_path = d / f"{tid}.arcs"

    track = load_track(track_path)
    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)

    return TripData(trip_id=tid, track=track, nodes=nodes, arcs=arcs)
