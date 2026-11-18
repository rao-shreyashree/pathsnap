"""Folium-based Interactive Geographical Route Visualisation.

Member D — `d-rf-hmm`
Generates multi-layer interactive HTML maps comparing:
1. Raw GPS observations (blue markers)
2. Road network geometry (grey lines)
3. Ground-truth reference route (emerald green)
4. Random Forest predicted route (dashed orange)
5. HMM/Viterbi decoded route (solid red)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from typing import Any, Sequence
import folium
import numpy as np
import pandas as pd

from src.config import MAPS_DIR
from src.road_graph import RoadGraph


def create_trip_map(
    trip_id: str,
    track_df: pd.DataFrame,
    road_graph: RoadGraph,
    ref_route_arcs: Sequence[int] | None = None,
    rf_pred_arcs: Sequence[int] | None = None,
    hmm_pred_arcs: Sequence[int] | None = None,
    output_html_path: str | Path | None = None,
) -> folium.Map:
    """Generate interactive Folium map visualizing vehicle trajectory and route reconstructions."""
    # Compute center of map from track
    center_lat = float(track_df["lat"].mean())
    center_lon = float(track_df["lon"].mean())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri WorldStreetMap",
        control_scale=True,
    )

    # 1. Road Network Layer (Background)
    net_group = folium.FeatureGroup(name="Road Network (OSM)", show=False)
    # Draw arcs in road graph
    for aid in road_graph._arc_ids:
        try:
            geom = road_graph.get_arc_geometry(int(aid))
            folium.PolyLine(
                locations=[
                    [geom["start_lat"], geom["start_lon"]],
                    [geom["end_lat"], geom["end_lon"]],
                ],
                color="#7f8c8d",
                weight=1.5,
                opacity=0.4,
                tooltip=f"Arc ID: {aid} (Len: {geom['length_m']:.1f}m)",
            ).add_to(net_group)
        except Exception:
            continue
    net_group.add_to(fmap)

    # 2. Raw GPS Observations Layer
    gps_group = folium.FeatureGroup(name="Raw GPS Fixes", show=True)
    lons = track_df["lon"].to_numpy()
    lats = track_df["lat"].to_numpy()
    ts = track_df["t"].to_numpy()

    # Subsample markers if track is very long (> 1000 points) to avoid browser lag
    step = max(1, len(lats) // 500)
    for i in range(0, len(lats), step):
        folium.CircleMarker(
            location=[float(lats[i]), float(lons[i])],
            radius=3.5,
            color="#2980b9",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.7,
            popup=f"Fix #{i}<br>Lat: {lats[i]:.6f}<br>Lon: {lons[i]:.6f}<br>t: {ts[i]:.1f}s",
        ).add_to(gps_group)

    # Add prominent START and END markers
    folium.Marker(
        location=[float(lats[0]), float(lons[0])],
        popup=f"<b>START OF TRIP</b><br>Fix #0<br>t: {ts[0]:.1f}s",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        tooltip="START (Fix #0)",
    ).add_to(gps_group)

    folium.Marker(
        location=[float(lats[-1]), float(lons[-1])],
        popup=f"<b>END OF TRIP</b><br>Fix #{len(lats)-1}<br>t: {ts[-1]:.1f}s",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        tooltip="END (Final Fix)",
    ).add_to(gps_group)

    gps_group.add_to(fmap)

    # Helper to build polyline coordinates from sequence of arc IDs
    def arcs_to_coords(arc_seq: Sequence[int]) -> list[list[float]]:
        coords: list[list[float]] = []
        for aid in arc_seq:
            if road_graph.is_valid_arc(aid):
                geom = road_graph.get_arc_geometry(aid)
                coords.append([geom["start_lat"], geom["start_lon"]])
                coords.append([geom["end_lat"], geom["end_lon"]])
        return coords

    # 3. Ground Truth Reference Route Layer
    if ref_route_arcs is not None and len(ref_route_arcs) > 0:
        ref_group = folium.FeatureGroup(name="Ground Truth Route", show=True)
        ref_coords = arcs_to_coords(ref_route_arcs)
        if ref_coords:
            folium.PolyLine(
                locations=ref_coords,
                color="#27ae60",  # Emerald Green
                weight=5.0,
                opacity=0.85,
                tooltip="Ground Truth Route (.route)",
            ).add_to(ref_group)
        ref_group.add_to(fmap)

    # 4. Random Forest Classifier-Only Route Layer
    if rf_pred_arcs is not None and len(rf_pred_arcs) > 0:
        rf_group = folium.FeatureGroup(name="RF Predicted Route (Classifier Only)", show=True)
        rf_coords = arcs_to_coords(rf_pred_arcs)
        if rf_coords:
            folium.PolyLine(
                locations=rf_coords,
                color="#e67e22",  # Amber Orange
                weight=3.5,
                opacity=0.85,
                dash_array="6, 6",
                tooltip="Random Forest Classifier Top-1 Predictions",
            ).add_to(rf_group)
        rf_group.add_to(fmap)

    # 5. HMM + Viterbi Decoded Route Layer
    if hmm_pred_arcs is not None and len(hmm_pred_arcs) > 0:
        hmm_group = folium.FeatureGroup(name="HMM + Viterbi Decoded Route", show=True)
        hmm_coords = arcs_to_coords(hmm_pred_arcs)
        if hmm_coords:
            folium.PolyLine(
                locations=hmm_coords,
                color="#c0392b",  # Dark Crimson Red
                weight=4.0,
                opacity=0.9,
                tooltip="HMM + Viterbi Decoded Route (Member D)",
            ).add_to(hmm_group)
        hmm_group.add_to(fmap)

    # Add HTML legend
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 290px; height: 200px; 
        background-color: white; border:2px solid #bdc3c7; z-index:9999; 
        font-size:12px; font-family: sans-serif; padding: 10px; border-radius: 6px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    ">
        <b style="font-size:13px; color:#2c3e50;">PathSnap Map: Trip {trip_id}</b><br>
        <span style="color:green; font-weight:bold;">&#9658;</span> <b>START</b> &nbsp;|&nbsp; <span style="color:red; font-weight:bold;">&#9873;</span> <b>END</b><br>
        <span style="color:#2980b9;">&#9679;</span> Raw GPS Fixes<br>
        <span style="color:#7f8c8d;">&#9472;</span> Road Network (OSM)<br>
        <span style="color:#27ae60; font-weight:bold;">&#9644;&#9644;</span> Ground Truth Route (.route)<br>
        <span style="color:#e67e22; font-weight:bold;">- - -</span> RF Classifier Only (Jittery)<br>
        <span style="color:#c0392b; font-weight:bold;">&#9644;&#9644;</span> RF + HMM + Viterbi (Decoded)
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Add Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    # Fit bounds
    min_lat = float(track_df["lat"].min())
    max_lat = float(track_df["lat"].max())
    min_lon = float(track_df["lon"].min())
    max_lon = float(track_df["lon"].max())
    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Export if requested
    if output_html_path is not None:
        p = Path(output_html_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(p))

    return fmap
