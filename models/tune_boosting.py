import json
import joblib
import numpy as np
import optuna
from pathlib import Path
from sklearn.model_selection import GroupKFold

from src.dataset import load_split, get_xy
from models.common import obs_top1, row_metrics
from models.train_boosting import make_model, SEED

OUT = Path("models/artifacts"); OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = 40
NEG_FRAC = 0.2   # speed: keep all positives + 20% of negatives when FITTING during search

train = load_split("train")
X, y, g = get_xy(train)
folds = list(GroupKFold(3).split(X, y, g))
rng = np.random.default_rng(SEED)


def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 500),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
        class_weight="balanced", random_state=SEED, early_stopping=False)
    scores = []
    for tr, va in folds:
        keep = (y.iloc[tr].values == 1) | (rng.random(len(tr)) < NEG_FRAC)
        m = make_model(params).fit(X.iloc[tr[keep]], y.iloc[tr[keep]])
        p = m.predict_proba(X.iloc[va])[:, 1]
        scores.append(obs_top1(train.iloc[va], p))   # validate on FULL fold
    return float(np.mean(scores))


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    study.trials_dataframe().to_csv(OUT / "optuna_trials.csv", index=False)
    best = {**study.best_params, "class_weight": "balanced", "random_state": SEED}
    json.dump({"best_value_cv_top1": study.best_value, "best_params": best},
              open(OUT / "boosting_best_params.json", "w"), indent=2)

    # refit on ALL 80 trips, evaluate once on student_test
    stest = load_split("student_test"); Xs, ys, _ = get_xy(stest)
    model = make_model(best).fit(X, y)
    ps = model.predict_proba(Xs)[:, 1]
    print({"student_top1": obs_top1(stest, ps), **row_metrics(ys, ps)})
    joblib.dump(model, OUT / "boosting_tuned.joblib")