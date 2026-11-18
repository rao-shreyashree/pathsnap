"""Helpers for teammates: load the processed tables.
    from src.dataset import load_split, get_xy
    df = load_split("train"); X, y, groups = get_xy(df)   # groups = trip_id for GroupKFold
"""
import pandas as pd
from . import config
from .candidates import FEATURE_COLS


def load_split(partition: str) -> pd.DataFrame:
    return pd.read_parquet(config.OUT_DIR / f"{partition}_features.parquet")


def load_obs(partition: str) -> pd.DataFrame:
    return pd.read_parquet(config.OUT_DIR / f"{partition}_obs.parquet")


def get_xy(df: pd.DataFrame):
    d = df[df.label >= 0]
    return d[FEATURE_COLS].to_numpy(float), d.label.to_numpy(), d.trip_id.to_numpy()
