"""Glue between A's candidate tables and B's evaluation harness (evaluate.py)."""
import sys

import pandas as pd

from src.candidates import FEATURE_COLS
from src.dataset import load_split

try:                       # B's file. Keep a copy in the repo root (or wherever B puts it).
    from evaluate import evaluate_model, evaluate_split, save_comparison  # noqa: F401
except ImportError:
    sys.exit("evaluate.py (from B) not found. Copy it into the repo root, then retry.")


def load_harness_frame(split: str) -> pd.DataFrame:
    """A's parquet -> the column names B's harness expects.

    reference_arc_id = the arc whose label is 1 for that GPS point (-1 if the true arc
    was not among the candidates = a candidate miss). from_node / to_node are dummies:
    A's tables do not store arc endpoints, so the harness's 'direct_arc_continuity'
    column is NOT meaningful for these runs. Our API computes real continuity instead.
    """
    df = load_split(split)
    df["trip_id"] = df["trip_id"].astype(str)
    df[FEATURE_COLS] = df[FEATURE_COLS].astype("float32")
    df = df.rename(columns={"obs_idx": "observation_id"})
    ref = (df.loc[df["label"] == 1].groupby(["trip_id", "observation_id"])["arc_id"]
             .first().rename("reference_arc_id"))
    df = df.join(ref, on=["trip_id", "observation_id"])
    df["reference_arc_id"] = df["reference_arc_id"].fillna(-1).astype("int64")
    df["from_node"], df["to_node"] = -1, -2
    return df


def obs_top1(df: pd.DataFrame, scores) -> float:
    """Top-1 per GPS point. Same tie-break as B's harness (highest score, then lowest arc_id).
    A point whose true arc is not a candidate counts as wrong."""
    d = pd.DataFrame({"trip_id": df["trip_id"].to_numpy(), "obs": df["observation_id"].to_numpy(),
                      "arc": df["arc_id"].to_numpy(), "label": df["label"].to_numpy(), "p": scores})
    win = d.sort_values(["p", "arc"], ascending=[False, True]).drop_duplicates(["trip_id", "obs"])
    return float(win["label"].mean())
