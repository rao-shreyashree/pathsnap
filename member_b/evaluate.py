"""Trip-grouped evaluation of candidate classifiers on development data."""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OBSERVATION = ["trip_id", "observation_id"]
METADATA = OBSERVATION + ["arc_id", "reference_arc_id", "from_node", "to_node"]
DEFAULT_FEATURES = ["distance_m", "heading_error_rad", "segment_fraction"]


def read_candidates(path, features):
    frame = pd.read_csv(path, dtype={"trip_id": str})
    required = METADATA + features
    missing = set(required) - set(frame.columns)
    if missing or frame.empty or frame[required].isna().any().any():
        raise ValueError(f"{path}: missing columns/values or empty table: {sorted(missing)}")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: features must be finite")
    if frame.duplicated(OBSERVATION + ["arc_id"]).any():
        raise ValueError(f"{path}: duplicate candidate per observation")
    if frame.groupby(OBSERVATION)["reference_arc_id"].nunique().gt(1).any():
        raise ValueError(f"{path}: each observation must have one reference arc")
    return frame


def evaluate_split(model, train, test, features):
    y_train = train.arc_id.eq(train.reference_arc_id).astype(int)
    y_test = test.arc_id.eq(test.reference_arc_id).astype(int)
    if y_train.nunique() != 2:
        raise ValueError("Each fitting split needs both positive and negative candidate labels")
    start = perf_counter()
    model.fit(train[features], y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    scores = model.predict_proba(test[features])[:, list(model.classes_).index(1)]
    inference_seconds = perf_counter() - start

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, scores > 0.5, labels=[0, 1], average="macro", zero_division=0
    )
    observed = test[METADATA].copy()
    observed["score"] = scores
    covered = y_test.groupby([test[col] for col in OBSERVATION]).any()
    ordered = observed.sort_values(["score", "arc_id"], ascending=[False, True])
    winners = ordered.drop_duplicates(OBSERVATION).sort_values(OBSERVATION)
    correct = winners.arc_id.eq(winners.reference_arc_id)
    previous = winners.groupby("trip_id")[["arc_id", "to_node"]].shift()
    transitions = previous.arc_id.notna()
    direct = winners.arc_id.eq(previous.arc_id) | winners.from_node.eq(previous.to_node)
    return {
        "trips": test.trip_id.nunique(),
        "observations": len(winners),
        "candidate_rows": len(test),
        "candidate_accuracy": accuracy_score(y_test, scores > 0.5),
        "candidate_macro_precision": precision,
        "candidate_macro_recall": recall,
        "candidate_macro_f1": macro_f1,
        "candidate_log_loss": log_loss(y_test, scores, labels=[0, 1]),
        "observation_top1": correct.mean(),
        "candidate_miss_rate": 1 - covered.mean(),
        "direct_arc_continuity": direct[transitions].mean(),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_observation": 1000 * inference_seconds / len(winners),
    }


def evaluate_model(factory, train, student_test, features, folds):
    """Run identical folds and scoring for any binary predict_proba classifier."""
    if set(train.trip_id) & set(student_test.trip_id):
        raise ValueError("Training and student-test trip IDs overlap")
    if not 2 <= folds <= train.trip_id.nunique():
        raise ValueError("Fold count must be between 2 and the number of training trips")
    rows = []
    groups = train.trip_id
    for number, (fit_indices, validation_indices) in enumerate(
        GroupKFold(n_splits=folds).split(train, groups=groups), start=1
    ):
        validation = train.iloc[validation_indices]
        result = evaluate_split(factory(), train.iloc[fit_indices], validation, features)
        rows.append({"split": f"cv_{number}", "trip_ids": "|".join(sorted(validation.trip_id.unique())), **result})
        print(f"CV fold {number}: macro F1 {result['candidate_macro_f1']:.4f}", flush=True)
    result = evaluate_split(factory(), train, student_test, features)
    rows.append({"split": "student_test", "trip_ids": "|".join(sorted(student_test.trip_id.unique())), **result})
    return pd.DataFrame(rows)


def save_comparison(results, output):
    """Accept one evaluation frame per model and write the common report table."""
    records = []
    for model_name, evaluations in results.items():
        cross_validation = evaluations[evaluations.split.str.startswith("cv_")]
        test = evaluations[evaluations.split.eq("student_test")].iloc[0]
        records.append({
            "model": model_name,
            "cv_macro_f1_mean": cross_validation.candidate_macro_f1.mean(),
            "cv_macro_f1_sd": cross_validation.candidate_macro_f1.std(),
            "student_top1": test.observation_top1,
            "student_macro_f1": test.candidate_macro_f1,
            "student_log_loss": test.candidate_log_loss,
            "student_candidate_miss_rate": test.candidate_miss_rate,
            "student_direct_arc_continuity": test.direct_arc_continuity,
            "student_ms_per_observation": test.inference_ms_per_observation,
        })
    summary = pd.DataFrame(records)
    summary.to_csv(output / "comparison.csv", index=False)
    panels = [
        ("cv_macro_f1_mean", "Grouped CV macro F1"),
        ("student_top1", "Student-test top-1"),
        ("student_macro_f1", "Student-test macro F1"),
        ("student_log_loss", "Student-test log loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for axis, (column, title) in zip(axes.flat, panels):
        bars = axis.barh(summary.model, summary[column], color="#176b87")
        axis.bar_label(bars, fmt=" %.3f", padding=3)
        axis.set_title(title, loc="left")
        maximum = 1.08 if column != "student_log_loss" else max(0.1, summary[column].max() * 1.25)
        axis.set_xlim(0, maximum)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    return summary


def knn(neighbors):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=neighbors, weights="distance", algorithm="kd_tree"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("student_test", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    args = parser.parse_args()
    if set(args.features) & set(METADATA):
        parser.error("Feature columns must exclude identifiers and reference labels")
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    train = read_candidates(args.train, args.features)
    test = read_candidates(args.student_test, args.features)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluations = evaluate_model(lambda: knn(args.neighbors), train, test, args.features, args.folds)
    evaluations.assign(model="Distance-weighted KNN", neighbors=args.neighbors,
                       features="|".join(args.features)).to_csv(args.output / "knn_metrics.csv", index=False)
    print(save_comparison({"Distance-weighted KNN": evaluations}, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
