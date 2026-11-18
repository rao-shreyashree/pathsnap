"""Folium map of a hard trip, highlighting GPS points whose reference arc is NOT among the candidates.
    python -m src.hard_trip_map                       # train trip with the most candidate misses
    python -m src.hard_trip_map --trip <trip_id> --partition train
Needs data/processed/{partition}_obs.parquet (run build_dataset first). Saves results/maps/hard_<trip_id>.html"""
import argparse
import numpy as np
import pandas as pd
import folium

from . import config
from .data_io import iter_trips
from .dataset import load_obs
from .folium_map import make_map, _arc_coords


def pick_trip(obs):
    lab = obs[obs.ref_in_cand >= 0]
    s = lab.groupby("trip_id").agg(n_obs=("obs_idx", "size"), misses=("ref_in_cand", lambda x: int((x == 0).sum())),
                                   ref_dist_p95=("ref_dist_m", lambda x: float(np.percentile(x, 95))))
    s["miss_rate"] = s.misses / s.n_obs
    s = s.sort_values(["misses", "ref_dist_p95"], ascending=False)
    print("Hardest trips:\n", s.head(5).round(3))
    return s.index[0], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="train"); ap.add_argument("--trip", default=None)
    args = ap.parse_args()
    obs = load_obs(args.partition)
    trip_id = args.trip or pick_trip(obs)[0]
    o = obs[obs.trip_id == trip_id]
    miss = o[o.ref_in_cand == 0]
    label = "Candidate misses (reference arc not within radius)"
    if miss.empty:                                    # no misses in this trip: show the noisiest points instead
        miss = o[o.ref_dist_m >= o.ref_dist_m.quantile(0.95)]
        label = "Noisiest GPS points (top 5% distance to reference)"
    print(f"{trip_id}: {len(o)} points, {int((o.ref_in_cand == 0).sum())} candidate misses, highlighting {len(miss)}")

    trip = next(t for t in iter_trips(args.partition) if t.trip_id == trip_id)
    m = make_map(trip)                                 # GPS + network + reference route, not saved yet

    g_miss = folium.FeatureGroup(label, show=True)
    for r in miss.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=6, color="#d62728", fill=True, fill_opacity=0.9,
                            tooltip=f"obs {r.obs_idx} | t={r.ts:.0f} | {r.ref_dist_m:.0f} m from reference arc").add_to(g_miss)
    g_miss.add_to(m)
    g_arc = folium.FeatureGroup("Reference arc of highlighted points", show=True)
    for seg in _arc_coords(trip, miss.ref_arc.unique()):
        folium.PolyLine(seg, color="#ff7f0e", weight=7, opacity=0.9).add_to(g_arc)
    g_arc.add_to(m)
    w = miss.loc[miss.ref_dist_m.idxmax()]             # worst point: draw the candidate search radius around it
    folium.Circle([w.lat, w.lon], radius=config.RADIUS_M, color="#d62728", weight=2, dash_array="6",
                  fill=False, tooltip=f"candidate search radius = {config.RADIUS_M:.0f} m").add_to(m)

    pad = 0.002                                        # zoom to the highlighted points
    m.fit_bounds([[miss.lat.min() - pad, miss.lon.min() - pad], [miss.lat.max() + pad, miss.lon.max() + pad]])
    config.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MAPS_DIR / f"hard_{trip_id}.html"
    m.save(str(out)); print("saved", out)


if __name__ == "__main__":
    main()
