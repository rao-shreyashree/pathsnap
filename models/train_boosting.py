"""Untuned HistGradientBoosting baseline, run through B's shared harness.

    python -m models.train_boosting
Outputs: results/histgb_untuned/*  and  models/artifacts/boosting_untuned.joblib
Config deliberately equals B's run_histgb.py defaults so the comparison table is consistent.
"""
import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from models.common import FEATURE_COLS, evaluate_model, load_harness_frame, save_comparison

SEED = 42
FOLDS = 4                       # same as B
UNTUNED = dict(max_iter=100, early_stopping=False, random_state=SEED)
OUT = Path("results/histgb_untuned")
ART = Path("models/artifacts")


def make_model(params):
    return HistGradientBoostingClassifier(**params)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    train, stest = load_harness_frame("train"), load_harness_frame("student_test")

    metrics = evaluate_model(lambda: make_model(UNTUNED), train, stest, FEATURE_COLS, FOLDS)
    metrics.assign(model="HistGradientBoosting (untuned)").to_csv(OUT / "histgb_metrics.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps(
        {"model_params": UNTUNED, "features": FEATURE_COLS, "folds": FOLDS}, indent=2))
    print(save_comparison({"HistGradientBoosting (untuned)": metrics}, OUT).to_string(index=False))

    t0 = time.time()
    model = make_model(UNTUNED).fit(train[FEATURE_COLS], train["label"])
    print(f"final fit on all 80 trips: {time.time() - t0:.1f}s")
    joblib.dump(model, ART / "boosting_untuned.joblib")
