"""Member A models: Logistic Regression + Gaussian Naive Bayes.
    python -m src.train_A                      # full run: 5-fold GroupKFold CV, fit on train, evaluate on student_test
    python -m src.train_A --sample-trips 8     # quick smoke test on 8 trips
    python -m src.train_A --skip-cv            # skip cross-validation
Outputs: experiments/A_results.csv, A_cv_folds.csv, A_config.json, figs/confusion_<model>.png, models/<model>.joblib
Primary metric = observation-level top-1 accuracy (per GPS point: highest-scoring candidate == reference arc;
GPS points whose reference arc is not among the candidates count as wrong)."""
import argparse, json, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (f1_score, precision_score, recall_score, log_loss,
                             confusion_matrix, ConfusionMatrixDisplay)

from . import config
from .dataset import load_split, load_obs
from .candidates import FEATURE_COLS

EXP_DIR = config.PROJECT_ROOT / "experiments"
MODEL_DIR = config.PROJECT_ROOT / "models"

HYPERPARAMS = {
    "logreg": dict(C=1.0, solver="lbfgs", max_iter=500, class_weight=None),
    "gnb": dict(var_smoothing=1e-9),
}


def make_model(name):
    if name == "logreg":
        clf = LogisticRegression(random_state=config.SEED, **HYPERPARAMS["logreg"])
    else:
        clf = GaussianNB(**HYPERPARAMS["gnb"])
    return make_pipeline(StandardScaler(), clf)


def load(partition, sample_trips=None):
    df, obs = load_split(partition), load_obs(partition)
    df = df[df.label >= 0]
    obs = obs[obs.ref_in_cand >= 0]                      # observations that have a reference arc
    if sample_trips:
        keep = sorted(df.trip_id.unique())[:sample_trips]
        df, obs = df[df.trip_id.isin(keep)], obs[obs.trip_id.isin(keep)]
    return df.reset_index(drop=True), obs.reset_index(drop=True)


def xyg(df):
    return df[FEATURE_COLS].to_numpy(np.float32), df.label.to_numpy(), df.trip_id.to_numpy()


def metrics(df, p, obs):
    y = df.label.to_numpy()
    d = df[["trip_id", "obs_idx", "label"]].copy()
    d["p"] = p
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False)["p"].idxmax(), "label"]
    pred = (p >= 0.5).astype(int)
    return dict(top1=float(best.sum() / max(len(obs), 1)),
                macro_f1=float(f1_score(y, pred, average="macro")),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                logloss=float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "gnb"], choices=["logreg", "gnb"])
    ap.add_argument("--sample-trips", type=int, default=None)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    for d in (EXP_DIR / "figs", MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    tr, tr_obs = load("train", args.sample_trips)
    te, te_obs = load("student_test", args.sample_trips)
    X, y, g = xyg(tr)
    Xte, yte, _ = xyg(te)
    print(f"train: {len(tr):,} rows, {tr.trip_id.nunique()} trips | test: {len(te):,} rows, {te.trip_id.nunique()} trips")

    summary, folds = [], []
    for name in args.models:
        print(f"\n=== {name} ===")
        row = dict(model=name, params=json.dumps(HYPERPARAMS[name]))
        if not args.skip_cv:
            gkf = GroupKFold(n_splits=min(5, tr.trip_id.nunique()))
            fm = []
            for k, (a, b) in enumerate(gkf.split(X, y, g)):
                m = make_model(name)
                t = time.time(); m.fit(X[a], y[a]); fit_s = time.time() - t
                va = tr.iloc[b]
                met = metrics(va, m.predict_proba(X[b])[:, 1], tr_obs[tr_obs.trip_id.isin(va.trip_id.unique())])
                met.update(model=name, fold=k, fit_seconds=fit_s)
                fm.append(met); folds.append(met)
                print(f"  fold {k}: top1={met['top1']:.4f} macroF1={met['macro_f1']:.4f} logloss={met['logloss']:.4f}")
            f = pd.DataFrame(fm)
            row.update(cv_top1_mean=f.top1.mean(), cv_top1_std=f.top1.std(), cv_macro_f1_mean=f.macro_f1.mean(),
                       cv_macro_f1_std=f.macro_f1.std(), cv_logloss_mean=f.logloss.mean())

        m = make_model(name)                                   # final fit on all 80 training trips
        t = time.time(); m.fit(X, y); row["train_seconds"] = time.time() - t
        for k, v in metrics(tr, m.predict_proba(X)[:, 1], tr_obs).items():   # for the over/underfitting question
            row[f"train_{k}"] = v
        t = time.time(); p = m.predict_proba(Xte)[:, 1]; row["infer_seconds"] = time.time() - t
        row["infer_ms_per_1k_rows"] = row["infer_seconds"] / len(te) * 1e6
        for k, v in metrics(te, p, te_obs).items():
            row[f"test_{k}"] = v
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k != "params"})

        cm = confusion_matrix(yte, (p >= 0.5).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["not ref arc", "ref arc"]).plot(values_format="d", cmap="Blues")
        plt.title(f"{name} - student_test (threshold 0.5)"); plt.tight_layout()
        plt.savefig(EXP_DIR / "figs" / f"confusion_{name}.png", dpi=150); plt.close()
        joblib.dump(m, MODEL_DIR / f"{name}.joblib")
        summary.append(row)

    pd.DataFrame(summary).to_csv(EXP_DIR / "A_results.csv", index=False)
    if folds:
        pd.DataFrame(folds).to_csv(EXP_DIR / "A_cv_folds.csv", index=False)
    json.dump(dict(seed=config.SEED, radius_m=config.RADIUS_M, max_cand=config.MAX_CAND, features=FEATURE_COLS,
                   hyperparams=HYPERPARAMS, cv="GroupKFold(5) by trip_id on the 80 training trips"),
              open(EXP_DIR / "A_config.json", "w"), indent=2)
    print(f"\nSaved results to {EXP_DIR} and models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
