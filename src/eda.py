"""Dataset exploration. python -m src.eda [--limit N]
Writes results/eda/trip_inventory.csv, radius_sweep.csv and PNG plots."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely

from . import config
from .data_io import iter_trips
from .candidates import project_trip, align_route
from .geo import haversine_m


def trip_stats(trip):
    tr, nd, ar = trip.track, trip.nodes, trip.arcs
    n = len(tr)
    ok = (np.isfinite(tr[["lon", "lat", "ts"]]).all(axis=1) & tr.lon.between(-180, 180) & tr.lat.between(-90, 90))
    ts = tr.ts.to_numpy(); dt = np.diff(ts) if n > 1 else np.array([np.nan])
    row = dict(partition=trip.partition, trip_id=trip.trip_id, n_obs=n, n_invalid=int((~ok).sum()),
               duration_s=float(np.nanmax(ts) - np.nanmin(ts)) if n else np.nan,
               dt_median=float(np.nanmedian(dt)), pct_dt_1s=float(np.mean(dt == 1) * 100),
               dt_max=float(np.nanmax(dt)), n_nonmonotonic=int((dt < 0).sum()),
               n_dup_ts=int((dt == 0).sum()), n_gaps_gt5s=int((dt > 5).sum()))
    v = tr[ok]
    row.update(lon_min=v.lon.min(), lon_max=v.lon.max(), lat_min=v.lat.min(), lat_max=v.lat.max(),
               extent_km=float(haversine_m(v.lon.min(), v.lat.min(), v.lon.max(), v.lat.max()) / 1000))
    # road network
    nn = len(nd); a, b = ar.from_node.to_numpy(), ar.to_node.to_numpy()
    good = (a >= 0) & (a < nn) & (b >= 0) & (b < nn)
    used = np.zeros(nn, bool); used[a[good]] = True; used[b[good]] = True
    ncomp = connected_components(coo_matrix((np.ones(good.sum()), (a[good], b[good])), shape=(nn, nn)),
                                 directed=False)[0] if nn else 0
    row.update(n_nodes=nn, n_arcs=len(ar), n_bad_arcs=int((~good).sum()), n_isolated_nodes=int((~used).sum()),
               n_components=int(ncomp - (~used).sum()))
    # reference route
    r = trip.route
    row["route_len"] = -1 if r is None else len(r)
    if r is not None and len(r):
        rv = r[(r >= 0) & (r < len(ar))]
        row["route_bad_ids"] = int(len(r) - len(rv))
        if len(rv) > 1:   # consecutive arcs should share a node: to_node[k] == from_node[k+1]
            row["route_disconnected_frac"] = float(np.mean(b[rv[:-1]] != a[rv[1:]]))
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    rows, dts, refd = [], [], []
    for part in config.PARTITIONS:
        for trip in iter_trips(part, args.limit):
            row = trip_stats(trip)
            try:
                g = project_trip(trip)
                pts = shapely.points(g.obs.x.to_numpy(), g.obs.y.to_numpy())
                al = align_route(g, trip.route, pts)
                if al is not None:
                    row.update(ref_dist_median=float(np.median(al[1])), ref_dist_p95=float(np.percentile(al[1], 95)),
                               ref_dist_max=float(al[1].max()))
                    refd.append(pd.DataFrame({"partition": part, "d": al[1]}))
                dts.append(pd.DataFrame({"partition": part, "dt": np.diff(g.obs.ts.to_numpy())}))
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)
    inv = pd.DataFrame(rows)
    inv.to_csv(config.EDA_DIR / "trip_inventory.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(inv.groupby("partition").agg(trips=("trip_id", "count"), obs_total=("n_obs", "sum"),
          obs_median=("n_obs", "median"), invalid=("n_invalid", "sum"), nonmono=("n_nonmonotonic", "sum"),
          gaps_gt5s=("n_gaps_gt5s", "sum"), dt_max=("dt_max", "max"), nodes_med=("n_nodes", "median"),
          arcs_med=("n_arcs", "median"), isolated=("n_isolated_nodes", "sum")))
    print("\nWorst trips by max time gap:\n", inv.nlargest(5, "dt_max")[["trip_id", "partition", "dt_max", "n_gaps_gt5s"]])

    if refd:   # candidate miss rate vs radius (ignores MAX_CAND truncation)
        rd = pd.concat(refd)
        sweep = pd.DataFrame({"radius_m": [20, 30, 50, 75, 100, 150, 200, 300]})
        sweep["miss_rate"] = [(rd.d > r).mean() for r in sweep.radius_m]
        sweep.to_csv(config.EDA_DIR / "radius_sweep.csv", index=False); print("\n", sweep)
        plt.figure(figsize=(6, 4)); plt.hist(rd.d.clip(upper=300), bins=60)
        plt.xlabel("GPS point -> reference arc distance (m, clipped at 300)"); plt.ylabel("count")
        plt.title("GPS noise relative to reference route"); plt.tight_layout()
        plt.savefig(config.EDA_DIR / "gps_to_route_dist.png", dpi=150); plt.close()
    if dts:
        d = pd.concat(dts); plt.figure(figsize=(6, 4))
        plt.hist(d.dt.clip(upper=30), bins=60, log=True); plt.xlabel("time between fixes (s, clipped at 30)")
        plt.ylabel("count (log)"); plt.title("Sampling interval"); plt.tight_layout()
        plt.savefig(config.EDA_DIR / "dt_hist.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 4))
    for part, grp in inv.groupby("partition"):
        plt.hist(grp.n_obs, bins=20, alpha=0.6, label=part)
    plt.xlabel("GPS points per trip"); plt.ylabel("trips"); plt.legend(); plt.tight_layout()
    plt.savefig(config.EDA_DIR / "obs_per_trip.png", dpi=150); plt.close()
    print(f"\nSaved to {config.EDA_DIR}")


if __name__ == "__main__":
    main()
