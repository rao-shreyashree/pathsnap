import json, time
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.dataset import load_split, get_xy
from models.common import obs_top1, row_metrics

SEED = 42
OUT = Path("models/artifacts"); OUT.mkdir(parents=True, exist_ok=True)

BASE_PARAMS = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31,
                   min_samples_leaf=20, l2_regularization=0.0,
                   class_weight="balanced", random_state=SEED)


def make_model(params):
    return HistGradientBoostingClassifier(**params)


if __name__ == "__main__":
    train, stest = load_split("train"), load_split("student_test")
    X, y, g = get_xy(train)
    Xs, ys, _ = get_xy(stest)

    cv = []
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, g)):
        m = make_model(BASE_PARAMS).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        r = row_metrics(y.iloc[va], p)
        r["top1"] = obs_top1(train.iloc[va], p)
        cv.append(r); print(f"fold {k}", r)

    t0 = time.time(); model = make_model(BASE_PARAMS).fit(X, y); train_s = time.time() - t0
    t0 = time.time(); ps = model.predict_proba(Xs)[:, 1]; inf_s = time.time() - t0
    res = {"params": BASE_PARAMS,
           "cv_macro_f1_mean": float(np.mean([r["macro_f1"] for r in cv])),
           "cv_macro_f1_std": float(np.std([r["macro_f1"] for r in cv])),
           "cv_top1_mean": float(np.mean([r["top1"] for r in cv])),
           "student_top1": obs_top1(stest, ps),
           **{f"student_{k}": v for k, v in row_metrics(ys, ps).items()},
           "train_seconds": train_s, "inference_seconds": inf_s}
    json.dump(res, open(OUT / "boosting_untuned.json", "w"), indent=2)
    joblib.dump(model, OUT / "boosting_untuned.joblib")
    print(res)