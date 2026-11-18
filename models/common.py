import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

ID_COLS = ["trip_id", "obs_idx", "arc_id"]


def obs_top1(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Top-1 accuracy per GPS point. Points whose true arc is not in the
    candidate set count as wrong (they never have a positive row)."""
    d = df[ID_COLS + ["label"]].reset_index(drop=True).copy()
    d["p"] = scores
    top = d.loc[d.groupby(["trip_id", "obs_idx"])["p"].idxmax()]
    return float(top["label"].mean())


def row_metrics(y, p, thr=0.5) -> dict:
    return {
        "macro_f1": float(f1_score(y, (p >= thr).astype(int), average="macro")),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }