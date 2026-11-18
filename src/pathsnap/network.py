"""Road network processing, spatial cropping, and topology indexing.

Implements:
- Spatial network cropping to track bounding box + protective margin.
- Equirectangular projection of network nodes.
- KDTree spatial indexing of sampled points along arc segments.
- Network adjacency graph representation for transition routing.
- Reverse twin detection (~91.8% of road segments have reverse twins).
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import scipy.sparse as sp

from src.pathsnap.geo import (
    project_equirectangular,
    compute_bearing,
    point_to_segment_distance,
)


class SpatialNetworkIndex(NamedTuple):
    tree: cKDTree
    sample_to_arc_idx: np.ndarray  # maps tree point index to index in arcs_df
    arc_sample_offsets: np.ndarray
    arc_x1: np.ndarray
    arc_y1: np.ndarray
    arc_x2: np.ndarray
    arc_y2: np.ndarray
    arc_lengths: np.ndarray
    arc_bearings: np.ndarray
    arc_ids: np.ndarray


def crop_network_to_bbox(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    margin_deg: float = 0.01,  # ~1.1 km margin
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """Crop road network nodes and arcs to a bounding box plus margin.
    
    Returns:
      cropped_nodes: filtered nodes DataFrame
      cropped_arcs: filtered arcs DataFrame (retaining original 'arc_id')
      node_id_map: mapping from original node_id to new contiguous 0-based node_id
    """
    lat_min = min_lat - margin_deg
    lat_max = max_lat + margin_deg
    lon_min = min_lon - margin_deg
    lon_max = max_lon + margin_deg

    node_lats = nodes_df["lat"].to_numpy()
    node_lons = nodes_df["lon"].to_numpy()
    node_mask = (
        (node_lats >= lat_min) & (node_lats <= lat_max) &
        (node_lons >= lon_min) & (node_lons <= lon_max)
    )

    valid_node_ids = set(np.where(node_mask)[0])

    from_nodes = arcs_df["from_node"].to_numpy()
    to_nodes = arcs_df["to_node"].to_numpy()

    # An arc is kept if at least one endpoint (or both) is within the crop zone
    arc_mask = [
        (u in valid_node_ids and v in valid_node_ids)
        for u, v in zip(from_nodes, to_nodes)
    ]
    arc_mask = np.array(arc_mask, dtype=bool)

    cropped_arcs = arcs_df[arc_mask].copy().reset_index(drop=True)

    # All nodes referenced by the kept arcs must be included
    kept_nodes_set = set(cropped_arcs["from_node"]).union(set(cropped_arcs["to_node"]))
    kept_nodes_sorted = sorted(kept_nodes_set)
    node_id_map = {orig_id: new_id for new_id, orig_id in enumerate(kept_nodes_sorted)}

    cropped_nodes = nodes_df.iloc[kept_nodes_sorted].copy().reset_index(drop=True)
    cropped_nodes["node_id"] = np.arange(len(cropped_nodes), dtype=np.int32)

    # Remap endpoints in arcs
    cropped_arcs["crop_from_node"] = cropped_arcs["from_node"].map(node_id_map).astype(np.int32)
    cropped_arcs["crop_to_node"] = cropped_arcs["to_node"].map(node_id_map).astype(np.int32)

    return cropped_nodes, cropped_arcs, node_id_map


def build_spatial_network_index(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    lon0: float,
    lat0: float,
    sample_spacing_m: float = 15.0,
) -> SpatialNetworkIndex:
    """Build a fast KDTree spatial index over uniformly sampled points on arc segments."""
    node_lons = nodes_df["lon"].to_numpy()
    node_lats = nodes_df["lat"].to_numpy()
    nx, ny = project_equirectangular(node_lons, node_lats, lon0, lat0)

    from_col = "crop_from_node" if "crop_from_node" in arcs_df.columns else "from_node"
    to_col = "crop_to_node" if "crop_to_node" in arcs_df.columns else "to_node"

    u = arcs_df[from_col].to_numpy()
    v = arcs_df[to_col].to_numpy()

    x1 = nx[u]
    y1 = ny[u]
    x2 = nx[v]
    y2 = ny[v]

    dx = x2 - x1
    dy = y2 - y1
    lengths = np.sqrt(dx * dx + dy * dy)
    bearings = compute_bearing(x1, y1, x2, y2)
    arc_ids = arcs_df["arc_id"].to_numpy()

    # Sample points along segments
    n_samples_per_arc = np.maximum(2, np.ceil(lengths / sample_spacing_m).astype(int))
    total_samples = int(np.sum(n_samples_per_arc))

    sample_pts = np.zeros((total_samples, 2), dtype=np.float64)
    sample_to_arc_idx = np.zeros(total_samples, dtype=np.int32)

    idx = 0
    for i in range(len(arcs_df)):
        k = n_samples_per_arc[i]
        alphas = np.linspace(0.0, 1.0, k, endpoint=True)
        sample_pts[idx : idx + k, 0] = x1[i] + alphas * dx[i]
        sample_pts[idx : idx + k, 1] = y1[i] + alphas * dy[i]
        sample_to_arc_idx[idx : idx + k] = i
        idx += k

    tree = cKDTree(sample_pts)

    return SpatialNetworkIndex(
        tree=tree,
        sample_to_arc_idx=sample_to_arc_idx,
        arc_sample_offsets=n_samples_per_arc,
        arc_x1=x1,
        arc_y1=y1,
        arc_x2=x2,
        arc_y2=y2,
        arc_lengths=lengths,
        arc_bearings=bearings,
        arc_ids=arc_ids,
    )


def find_reverse_twin_arcs(arcs_df: pd.DataFrame) -> np.ndarray:
    """Identify which arcs possess a reverse twin (v -> u).
    
    Returns:
      has_reverse_twin: boolean array of length len(arcs_df)
    """
    u = arcs_df["from_node"].to_numpy()
    v = arcs_df["to_node"].to_numpy()
    edge_set = set(zip(u, v))
    has_twin = np.array([(v_i, u_i) in edge_set for u_i, v_i in zip(u, v)], dtype=bool)
    return has_twin


def compute_node_degrees(n_nodes: int, arcs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute in-degree and out-degree for each node."""
    from_col = "crop_from_node" if "crop_from_node" in arcs_df.columns else "from_node"
    to_col = "crop_to_node" if "crop_to_node" in arcs_df.columns else "to_node"

    u = arcs_df[from_col].to_numpy()
    v = arcs_df[to_col].to_numpy()

    out_deg = np.bincount(u, minlength=n_nodes)
    in_deg = np.bincount(v, minlength=n_nodes)
    return in_deg, out_deg


def build_adjacency_graph(
    n_nodes: int, arcs_df: pd.DataFrame
) -> tuple[sp.csr_matrix, dict[tuple[int, int], int]]:
    """Build directed adjacency graph with edge weights equal to arc lengths."""
    from_col = "crop_from_node" if "crop_from_node" in arcs_df.columns else "from_node"
    to_col = "crop_to_node" if "crop_to_node" in arcs_df.columns else "to_node"

    u = arcs_df[from_col].to_numpy()
    v = arcs_df[to_col].to_numpy()

    if "length_m" in arcs_df.columns:
        weights = arcs_df["length_m"].to_numpy()
    else:
        weights = np.ones(len(arcs_df), dtype=np.float64)

    adj = sp.csr_matrix((weights, (u, v)), shape=(n_nodes, n_nodes))
    edge_to_arc = {(int(u[i]), int(v[i])): int(arcs_df["arc_id"].iloc[i]) for i in range(len(arcs_df))}

    return adj, edge_to_arc
