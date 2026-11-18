"""Build candidate/feature tables.
    python -m src.build_dataset --partitions train student_test [--radius 100] [--max-cand 20] [--limit 5]
Outputs data/processed/{partition}_features.parquet, {partition}_obs.parquet, meta.json"""
import argparse, json
import pandas as pd
from tqdm import tqdm

from . import config
from .data_io import iter_trips
from .candidates import build_trip_tables, FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=config.PARTITIONS)
    ap.add_argument("--radius", type=float, default=config.RADIUS_M)
    ap.add_argument("--max-cand", type=int, default=config.MAX_CAND)
    ap.add_argument("--limit", type=int, default=None, help="only first N trips (debug)")
    args = ap.parse_args()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    for part in args.partitions:
        obs_all, cand_all = [], []
        for trip in tqdm(iter_trips(part, args.limit), desc=part):
            try:
                o, c = build_trip_tables(trip, args.radius, args.max_cand)
            except Exception as e:
                print(f"  ! skipped {trip.trip_id}: {e}")
                continue
            obs_all.append(o); cand_all.append(c)
        obs, cand = pd.concat(obs_all, ignore_index=True), pd.concat(cand_all, ignore_index=True)
        obs.to_parquet(config.OUT_DIR / f"{part}_obs.parquet", index=False)
        cand.to_parquet(config.OUT_DIR / f"{part}_features.parquet", index=False)
        lab = obs[obs.ref_in_cand >= 0]
        print(f"[{part}] trips={obs.trip_id.nunique()} obs={len(obs)} candidates={len(cand)} "
              f"cand/obs={len(cand)/len(obs):.1f}")
        if len(lab):
            print(f"[{part}] candidate miss rate = {1 - lab.ref_in_cand.mean():.4f} | "
                  f"positive-label rate = {(cand.label == 1).mean():.4f} | "
                  f"median ref dist = {lab.ref_dist_m.median():.1f} m")
    json.dump({"radius_m": args.radius, "max_cand": args.max_cand, "seed": config.SEED,
               "feature_cols": FEATURE_COLS}, open(config.OUT_DIR / "meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
