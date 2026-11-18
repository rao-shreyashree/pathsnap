"""Read raw PathSnap files: .track (lon lat ts), .nodes (lon lat), .arcs (from to), .route (arc ids)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # lon, lat, ts  (row order = file order)
    nodes: pd.DataFrame          # lon, lat      (row index = node id)
    arcs: pd.DataFrame           # from_node, to_node (row index = arc id)
    route: np.ndarray | None     # reference arc ids in travel order, or None


def _read(path, names, as_int=False):
    if path is None:
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=range(len(names)),
                         names=names, dtype="float64")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame({n: pd.Series(dtype="float64") for n in names})
    if as_int:   # unparsable / missing -> -1
        df = pd.DataFrame({c: np.where(np.isfinite(df[c]), df[c], -1).astype("int64") for c in names})
    return df


def _sibling(track: Path, ext: str):
    p = track.with_suffix(ext)
    if p.exists():
        return p
    found = sorted(track.parent.glob(f"*{ext}"))
    return found[0] if len(found) == 1 else None


def load_trip(track_path: Path, partition: str, trip_id: str | None = None) -> Trip:
    track_path = Path(track_path)
    tid = trip_id or track_path.stem
    nodes_p, arcs_p, route_p = (_sibling(track_path, e) for e in (".nodes", ".arcs", ".route"))
    if nodes_p is None or arcs_p is None:
        raise FileNotFoundError(f"{tid}: could not find .nodes/.arcs next to {track_path}")
    route = _read(route_p, ["arc"], as_int=True)
    return Trip(
        trip_id=tid, partition=partition,
        track=_read(track_path, ["lon", "lat", "ts"]),
        nodes=_read(nodes_p, ["lon", "lat"]),
        arcs=_read(arcs_p, ["from_node", "to_node"], as_int=True),
        route=None if route is None else route["arc"].to_numpy(),
    )


def iter_trips(partition: str, limit: int | None = None):
    pdir = config.DATA_ROOT / partition
    if not pdir.is_dir():
        have = [p.name for p in config.DATA_ROOT.iterdir()] if config.DATA_ROOT.is_dir() else "DATA_ROOT missing"
        raise FileNotFoundError(f"{pdir} not found. DATA_ROOT contains: {have}")
    tracks = sorted(pdir.rglob("*.track"))
    if not tracks:
        raise FileNotFoundError(f"No .track files under {pdir}")
    stems = [t.stem for t in tracks]
    dup = {s for s in stems if stems.count(s) > 1}
    for t in tracks[:limit]:
        tid = t.stem if t.stem not in dup else f"{t.parent.name}_{t.stem}"
        yield load_trip(t, partition, tid)
