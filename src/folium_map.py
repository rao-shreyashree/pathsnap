"""Interactive Folium map of one trip: GPS, road network, reference route, (optional) predicted route.
    python -m src.folium_map --partition train [--trip TRIP_ID]
D/B can reuse:  make_map(trip, "out.html", pred_arcs=[arc ids in order])"""
import argparse
import folium
import numpy as np

from . import config
from .data_io import iter_trips

LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;
border:1px solid #888;font-size:13px"><b>Legend</b><br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS points<br>
<span style="color:#2ca02c">&#9644;</span> Reference route<br>
<span style="color:#d62728">&#9644;</span> Predicted route<br>
<span style="color:#999">&#9644;</span> Road network</div>"""


def _arc_coords(trip, arc_ids):
    nd, ar = trip.nodes, trip.arcs
    out = []
    for k in arc_ids:
        if 0 <= k < len(ar):
            a, b = ar.from_node.iat[k], ar.to_node.iat[k]
            if 0 <= a < len(nd) and 0 <= b < len(nd):
                out.append([[nd.lat.iat[a], nd.lon.iat[a]], [nd.lat.iat[b], nd.lon.iat[b]]])
    return out


def make_map(trip, out_html, pred_arcs=None, margin_deg=0.004, max_net_arcs=8000):
    tr = trip.track.dropna()
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])

    # road network, restricted to the trip's neighbourhood
    nd, ar = trip.nodes, trip.arcs
    lat, lon = nd.lat.to_numpy(), nd.lon.to_numpy()
    near = ((lat >= tr.lat.min() - margin_deg) & (lat <= tr.lat.max() + margin_deg) &
            (lon >= tr.lon.min() - margin_deg) & (lon <= tr.lon.max() + margin_deg))
    a, b = ar.from_node.to_numpy(), ar.to_node.to_numpy()
    ok = (a >= 0) & (a < len(nd)) & (b >= 0) & (b < len(nd))
    sel = np.flatnonzero(ok)
    sel = sel[near[a[sel]] | near[b[sel]]]
    if len(sel) > max_net_arcs:
        sel = np.random.default_rng(config.SEED).choice(sel, max_net_arcs, replace=False)
    g_net = folium.FeatureGroup("Road network", show=True)
    for seg in _arc_coords(trip, sel):
        folium.PolyLine(seg, color="#999", weight=2, opacity=0.6).add_to(g_net)
    g_net.add_to(m)

    g_gps = folium.FeatureGroup("Raw GPS", show=True)
    for i, r in enumerate(tr.itertuples()):
        folium.CircleMarker([r.lat, r.lon], radius=3, color="#1f77b4", fill=True, fill_opacity=0.8,
                            tooltip=f"row {i} | t={r.ts:.0f}").add_to(g_gps)
    g_gps.add_to(m)

    layers = [("Reference route", trip.route, "#2ca02c", 6), ("Predicted route", pred_arcs, "#d62728", 4)]
    for name, arcs, color, w in layers:
        if arcs is None or len(arcs) == 0:
            continue
        g = folium.FeatureGroup(name, show=True)
        for seg in _arc_coords(trip, arcs):
            folium.PolyLine(seg, color=color, weight=w, opacity=0.9).add_to(g)
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND))
    m.save(str(out_html))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="train"); ap.add_argument("--trip", default=None)
    args = ap.parse_args()
    config.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    for t in iter_trips(args.partition):
        if args.trip is None or t.trip_id == args.trip:
            out = config.MAPS_DIR / f"{t.trip_id}.html"
            make_map(t, out); print("saved", out); break
