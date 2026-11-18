"""Bayesian hyperparameter optimisation (Optuna TPE + median pruning) for HistGradientBoosting.

    python -m models.tune_boosting --trials 25
Objective : maximise mean observation-level top-1 over trip-grouped CV (80 training trips only).
Search record: models/artifacts/optuna.db (resumable) and optuna_trials.csv.
The student-test trips are used ONCE at the end; evaluator trips are never touched.
Speed trick: while searching, each fold's FIT uses all positives + a random --neg-frac of
negatives; validation always uses the full fold. The final refit uses all rows.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from models.common import (FEATURE_COLS, evaluate_split, load_harness_frame, obs_top1)
from models.train_boosting import FOLDS, SEED, make_model

ART = Path("models/artifacts")
RES = Path("results/histgb_tuned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--neg-frac", type=float, default=0.25)
    ap.add_argument("--timeout", type=int, default=None, help="seconds, optional")
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    train = load_harness_frame("train")
    X, y = train[FEATURE_COLS], train["label"].to_numpy()
    folds = list(GroupKFold(n_splits=FOLDS).split(train, groups=train["trip_id"]))
    rng = np.random.default_rng(SEED)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 400),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 500, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            early_stopping=False, random_state=SEED)
        scores = []
        for step, (tr, va) in enumerate(folds):
            keep = (y[tr] == 1) | (rng.random(len(tr)) < args.neg_frac)
            m = make_model(params).fit(X.iloc[tr[keep]], y[tr[keep]])
            p = m.predict_proba(X.iloc[va])[:, list(m.classes_).index(1)]
            scores.append(obs_top1(train.iloc[va], p))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", study_name="histgb_top1",
        storage=f"sqlite:///{(ART / 'optuna.db').as_posix()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    search_seconds = time.time() - t0
    study.trials_dataframe().to_csv(ART / "optuna_trials.csv", index=False)

    best = {**study.best_params, "early_stopping": False, "random_state": SEED}
    json.dump({"objective": "mean observation top-1, grouped CV", "folds": FOLDS,
               "best_cv_top1": study.best_value, "best_params": best,
               "n_trials": len(study.trials), "search_seconds_this_run": search_seconds,
               "neg_frac_during_search": args.neg_frac},
              open(ART / "boosting_best_params.json", "w"), indent=2)
    print("best:", study.best_value, best)

    # Refit on ALL 80 trips, evaluate ONCE on the 10 student-test trips.
    stest = load_harness_frame("student_test")
    model = make_model(best)
    res = evaluate_split(model, train, stest, FEATURE_COLS)     # fits `model` on train
    joblib.dump(model, ART / "boosting_tuned.joblib")
    pd.DataFrame([{"split": "student_test", **res}]).to_csv(RES / "tuned_student_metrics.csv", index=False)

    untuned = Path("results/histgb_untuned/histgb_metrics.csv")
    if untuned.exists():
        u = pd.read_csv(untuned).query("split == 'student_test'").iloc[0]
        keys = ["observation_top1", "candidate_macro_f1", "candidate_log_loss",
                "fit_seconds", "inference_ms_per_observation"]
        pd.DataFrame({"metric": keys, "untuned": [u[k] for k in keys],
                      "tuned": [res[k] for k in keys]}).to_csv(RES / "tuned_vs_untuned.csv", index=False)
    print({k: res[k] for k in ["observation_top1", "candidate_macro_f1", "candidate_log_loss"]})


if __name__ == "__main__":
    main()
