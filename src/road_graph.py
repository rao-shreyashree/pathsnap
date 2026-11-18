"""Road network graph construction, topological indexing, and distance calculations.

Member D — `d-rf-hmm`
Preserves original zero-based node and arc IDs, constructs directed connectivity,
and provides ultra-fast successor and shortest-path lookups for HMM decoding.
"""
from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import (
    EARTH_RADIUS_M,
    DEFAULT_CROP_MARGIN_DEG,
    SAMPLE_SPACING_M,
)


def project_equirectangular(
    lons: np.ndarray,
    lats: np.ndarray,
    lon0: float,
    lat0: float,
    radius: float = EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Project spherical coordinates (lon, lat) in degrees to metric Cartesian (x, y) in metres."""
    lat0_rad = np.radians(lat0)
    dlon_rad = np.radians(lons - lon0)
    dlat_rad = np.radians(lats - lat0)
    x = radius * dlon_rad * np.cos(lat0_rad)
    y = radius * dlat_rad
    return x, y


def compute_bearing(
    x1: float | np.ndarray,
    y1: float | np.ndarray,
    x2: float | np.ndarray,
    y2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute direction angle in degrees [0, 360) from (x1, y1) to (x2, y2).
    
    0 deg = North (+y), 90 deg = East (+x), 180 deg = South (-y), 270 deg = West (-x).
    """
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dx, dy))
    return (angle + 360.0) % 360.0


def point_to_segment_distance(
    px: float | np.ndarray,
    py: float | np.ndarray,
    x1: float | np.ndarray,
    y1: float | np.ndarray,
    x2: float | np.ndarray,
    y2: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate perpendicular distance from point P to segment S((x1,y1) -> (x2, y2)).
    
    Returns:
      dist: distance in metres
      t: fractional projection along segment [0, 1]
      qx, qy: projected coordinate on segment
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    zero_len = seg_len_sq < 1e-12
    seg_len_sq_safe = np.where(zero_len, 1.0, seg_len_sq)

    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq_safe
    t = np.clip(t, 0.0, 1.0)
    t = np.where(zero_len, 0.0, t)

    qx = x1 + t * dx
    qy = y1 + t * dy
    dist = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
    return dist, t, qx, qy


class RoadGraph:
    """Directed Road Network Graph for map-matching and sequence decoding.
    
    Attributes:
      nodes_df: DataFrame with ['node_id', 'lon', 'lat', 'x', 'y']
      arcs_df: DataFrame with ['arc_id', 'from_node', 'to_node', 'length_m', 'bearing_deg']
      lon0, lat0: Geographic projection centroid
      successors: Mapping arc_id -> list of immediately connected succeeding arc_ids
      predecessors: Mapping arc_id -> list of immediately connected preceding arc_ids
      reverse_twin: Mapping arc_id -> reverse twin arc_id (if exists)
    """

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        lon0: float,
        lat0: float,
    ) -> None:
        self.nodes_df = nodes_df.copy()
        self.arcs_df = arcs_df.copy()
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)

        # Ensure projected metric coordinates (x, y) exist on nodes
        if "x" not in self.nodes_df.columns or "y" not in self.nodes_df.columns:
            nx, ny = project_equirectangular(
                self.nodes_df["lon"].to_numpy(),
                self.nodes_df["lat"].to_numpy(),
                self.lon0,
                self.lat0,
            )
            self.nodes_df["x"] = nx
            self.nodes_df["y"] = ny

        # Build fast NumPy arrays for coordinate lookups
        self._node_x = self.nodes_df["x"].to_numpy()
        self._node_y = self.nodes_df["y"].to_numpy()
        self._node_lons = self.nodes_df["lon"].to_numpy()
        self._node_lats = self.nodes_df["lat"].to_numpy()

        # Build fast arc attributes
        u = self.arcs_df["from_node"].to_numpy()
        v = self.arcs_df["to_node"].to_numpy()
        self._from_nodes = u
        self._to_nodes = v

        x1 = self._node_x[u]
        y1 = self._node_y[u]
        x2 = self._node_x[v]
        y2 = self._node_y[v]

        dx = x2 - x1
        dy = y2 - y1
        lengths = np.sqrt(dx * dx + dy * dy)
        bearings = compute_bearing(x1, y1, x2, y2)

        self.arcs_df["length_m"] = lengths
        self.arcs_df["bearing_deg"] = bearings

        self._arc_x1 = x1
        self._arc_y1 = y1
        self._arc_x2 = x2
        self._arc_y2 = y2
        self._arc_lengths = lengths
        self._arc_bearings = bearings
        self._arc_ids = self.arcs_df["arc_id"].to_numpy()

        # Map original arc_id to internal row index
        self._arc_id_to_idx = {int(aid): idx for idx, aid in enumerate(self._arc_ids)}

        # Build node to outgoing/incoming arcs lookups
        n_nodes = len(self.nodes_df)
        self.node_out_arcs: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        self.node_in_arcs: dict[int, list[int]] = {i: [] for i in range(n_nodes)}

        for idx, aid in enumerate(self._arc_ids):
            u_node = int(u[idx])
            v_node = int(v[idx])
            self.node_out_arcs[u_node].append(int(aid))
            self.node_in_arcs[v_node].append(int(aid))

        # Build arc successors and predecessors lookups
        # An arc A (u -> v) has successor B (v -> w) iff to_node(A) == from_node(B)
        self.successors: dict[int, list[int]] = {}
        self.predecessors: dict[int, list[int]] = {}

        for idx, aid in enumerate(self._arc_ids):
            aid_int = int(aid)
            v_node = int(v[idx])
            u_node = int(u[idx])
            self.successors[aid_int] = list(self.node_out_arcs[v_node])
            self.predecessors[aid_int] = list(self.node_in_arcs[u_node])

        # Reverse twin detection (v -> u)
        edge_to_arc = {}
        for idx, aid in enumerate(self._arc_ids):
            edge_to_arc[(int(u[idx]), int(v[idx]))] = int(aid)

        self.reverse_twin: dict[int, int | None] = {}
        for idx, aid in enumerate(self._arc_ids):
            aid_int = int(aid)
            rev_edge = (int(v[idx]), int(u[idx]))
            self.reverse_twin[aid_int] = edge_to_arc.get(rev_edge, None)

        # Distance lookup cache for HMM transitions: (arc_u, arc_v) -> distance
        self._dist_cache: dict[tuple[int, int], float | None] = {}

    @property
    def num_nodes(self) -> int:
        return len(self.nodes_df)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs_df)

    def is_valid_arc(self, arc_id: int) -> bool:
        """Check if an arc_id exists in this road graph."""
        return arc_id in self._arc_id_to_idx

    def get_arc_geometry(self, arc_id: int) -> dict[str, Any]:
        """Return metric and geographic geometry for an arc."""
        if not self.is_valid_arc(arc_id):
            raise KeyError(f"Arc ID {arc_id} not found in graph")
        idx = self._arc_id_to_idx[arc_id]
        u = self._from_nodes[idx]
        v = self._to_nodes[idx]
        return {
            "arc_id": arc_id,
            "from_node": int(u),
            "to_node": int(v),
            "start_lon": float(self._node_lons[u]),
            "start_lat": float(self._node_lats[u]),
            "end_lon": float(self._node_lons[v]),
            "end_lat": float(self._node_lats[v]),
            "x1": float(self._arc_x1[idx]),
            "y1": float(self._arc_y1[idx]),
            "x2": float(self._arc_x2[idx]),
            "y2": float(self._arc_y2[idx]),
            "length_m": float(self._arc_lengths[idx]),
            "bearing_deg": float(self._arc_bearings[idx]),
            "reverse_twin": self.reverse_twin[arc_id],
        }

    def is_valid_transition(self, arc_u: int, arc_v: int) -> bool:
        """Check if arc_v is an immediate topological successor of arc_u."""
        if not self.is_valid_arc(arc_u) or not self.is_valid_arc(arc_v):
            return False
        return arc_v in self.successors.get(arc_u, [])

    def shortest_network_distance(
        self,
        arc_u: int,
        arc_v: int,
        max_dist_m: float = 1000.0,
    ) -> float | None:
        """Calculate shortest directed road-network distance from end of arc_u to start of arc_v.
        
        Returns:
          0.0 if arc_u == arc_v or arc_v is immediate successor (to_node(u) == from_node(v))
          float >= 0.0 representing cumulative traversed arc lengths along the path
          None if no directed path exists within max_dist_m
        """
        if arc_u == arc_v:
            return 0.0

        if not self.is_valid_arc(arc_u) or not self.is_valid_arc(arc_v):
            return None

        # Check immediate successor (0 intermediate distance)
        if arc_v in self.successors.get(arc_u, []):
            return 0.0

        cache_key = (arc_u, arc_v)
        if cache_key in self._dist_cache:
            return self._dist_cache[cache_key]

        # Targeted Dijkstra over arc adjacency graph
        # State in queue: (dist_so_far, current_arc_id)
        pq: list[tuple[float, int]] = []
        visited: dict[int, float] = {}

        # Initialise with immediate successors of arc_u
        for next_arc in self.successors.get(arc_u, []):
            next_idx = self._arc_id_to_idx[next_arc]
            next_len = self._arc_lengths[next_idx]
            if next_len <= max_dist_m:
                heapq.heappush(pq, (next_len, next_arc))
                visited[next_arc] = next_len

        found_dist: float | None = None

        while pq:
            d_curr, curr_arc = heapq.heappop(pq)

            if curr_arc == arc_v:
                # Reached target arc! The distance to reach arc_v's start is d_curr minus arc_v's length
                idx_v = self._arc_id_to_idx[arc_v]
                found_dist = max(0.0, d_curr - self._arc_lengths[idx_v])
                break

            if d_curr > visited.get(curr_arc, float("inf")):
                continue

            for nxt_arc in self.successors.get(curr_arc, []):
                nxt_idx = self._arc_id_to_idx[nxt_arc]
                nxt_len = self._arc_lengths[nxt_idx]
                d_new = d_curr + nxt_len

                if d_new <= max_dist_m and d_new < visited.get(nxt_arc, float("inf")):
                    visited[nxt_arc] = d_new
                    heapq.heappush(pq, (d_new, nxt_arc))

        self._dist_cache[cache_key] = found_dist
        return found_dist

    def validate_integrity(self) -> dict[str, Any]:
        """Audit graph structure and report topological health."""
        n_nodes = len(self.nodes_df)
        n_arcs = len(self.arcs_df)

        invalid_from = np.sum((self._from_nodes < 0) | (self._from_nodes >= n_nodes))
        invalid_to = np.sum((self._to_nodes < 0) | (self._to_nodes >= n_nodes))
        self_loops = np.sum(self._from_nodes == self._to_nodes)

        # Isolated nodes (0 in-degree and 0 out-degree)
        in_degrees = np.bincount(self._to_nodes, minlength=n_nodes)
        out_degrees = np.bincount(self._from_nodes, minlength=n_nodes)
        isolated_nodes = np.sum((in_degrees == 0) & (out_degrees == 0))

        # Twin share
        n_twins = sum(1 for t in self.reverse_twin.values() if t is not None)
        twin_pct = (n_twins / n_arcs * 100.0) if n_arcs > 0 else 0.0

        return {
            "num_nodes": n_nodes,
            "num_arcs": n_arcs,
            "invalid_from_nodes": int(invalid_from),
            "invalid_to_nodes": int(invalid_to),
            "self_loops": int(self_loops),
            "isolated_nodes": int(isolated_nodes),
            "arcs_with_reverse_twins": n_twins,
            "reverse_twin_percentage": round(twin_pct, 2),
        }

    @classmethod
    def from_files(
        cls,
        nodes_path: str | Path,
        arcs_path: str | Path,
        bbox: tuple[float, float, float, float] | None = None,
        margin_deg: float = DEFAULT_CROP_MARGIN_DEG,
    ) -> RoadGraph:
        """Load RoadGraph from raw .nodes and .arcs files with optional bounding box crop.
        
        bbox format: (min_lon, min_lat, max_lon, max_lat)
        """
        nodes_df = pd.read_csv(
            nodes_path,
            sep=r"\s+",
            header=None,
            names=["lon", "lat"],
            dtype={"lon": np.float64, "lat": np.float64},
            engine="c",
        )
        nodes_df["node_id"] = np.arange(len(nodes_df), dtype=np.int32)

        arcs_df = pd.read_csv(
            arcs_path,
            sep=r"\s+",
            header=None,
            names=["from_node", "to_node"],
            dtype={"from_node": np.int32, "to_node": np.int32},
            engine="c",
        )
        arcs_df["arc_id"] = np.arange(len(arcs_df), dtype=np.int32)

        # Centroid for projection
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            lon0 = (min_lon + max_lon) / 2.0
            lat0 = (min_lat + max_lat) / 2.0
            
            # Crop nodes and arcs to bbox + margin
            lat_min = min_lat - margin_deg
            lat_max = max_lat + margin_deg
            lon_min = min_lon - margin_deg
            lon_max = max_lon + margin_deg

            node_mask = (
                (nodes_df["lat"] >= lat_min) & (nodes_df["lat"] <= lat_max) &
                (nodes_df["lon"] >= lon_min) & (nodes_df["lon"] <= lon_max)
            )
            valid_node_ids = set(np.where(node_mask)[0])

            u = arcs_df["from_node"].to_numpy()
            v = arcs_df["to_node"].to_numpy()
            arc_mask = [(u_i in valid_node_ids and v_i in valid_node_ids) for u_i, v_i in zip(u, v)]
            arcs_df = arcs_df[arc_mask].copy().reset_index(drop=True)

            # Keep all nodes referenced by surviving arcs
            kept_nodes = sorted(set(arcs_df["from_node"]).union(set(arcs_df["to_node"])))
            node_map = {orig: new for new, orig in enumerate(kept_nodes)}

            nodes_df = nodes_df.iloc[kept_nodes].copy().reset_index(drop=True)
            nodes_df["orig_node_id"] = nodes_df["node_id"]
            nodes_df["node_id"] = np.arange(len(nodes_df), dtype=np.int32)

            arcs_df["from_node"] = arcs_df["from_node"].map(node_map).astype(np.int32)
            arcs_df["to_node"] = arcs_df["to_node"].map(node_map).astype(np.int32)
        else:
            lon0 = float(nodes_df["lon"].mean())
            lat0 = float(nodes_df["lat"].mean())

        return cls(nodes_df=nodes_df, arcs_df=arcs_df, lon0=lon0, lat0=lat0)

    @classmethod
    def from_trip_directory(
        cls,
        trip_dir: str | Path,
        crop: bool = True,
        margin_deg: float = DEFAULT_CROP_MARGIN_DEG,
    ) -> RoadGraph:
        """Load RoadGraph for a trip directory containing <trip_id>.nodes and .arcs."""
        d = Path(trip_dir)
        tid = d.name
        nodes_path = d / f"{tid}.nodes"
        arcs_path = d / f"{tid}.arcs"
        track_path = d / f"{tid}.track"

        bbox = None
        if crop and track_path.exists():
            track_df = pd.read_csv(
                track_path,
                sep=r"\s+",
                header=None,
                names=["lon", "lat", "t"],
                engine="c",
            )
            bbox = (
                float(track_df["lon"].min()),
                float(track_df["lat"].min()),
                float(track_df["lon"].max()),
                float(track_df["lat"].max()),
            )

        return cls.from_files(nodes_path, arcs_path, bbox=bbox, margin_deg=margin_deg)
