# PathSnap handoff from B to C

## Connect HistGradientBoosting

Keep run_histgb.py beside evaluate.py. Install dependencies with python -m pip install -r requirements.txt. Run from the directory containing both scripts.

~~~powershell
python run_histgb.py candidate_data/train.csv candidate_data/student_test.csv results/histgb --folds 4
~~~

Use Person A's agreed feature columns and pass them in order through --features if they differ from the prototype columns distance_m, heading_error_rad and segment_fraction. Use identical candidate tables, features and fold count for every model you intend to compare.

The harness takes a factory that returns a fresh binary classifier supporting fit, predict_proba and classes_. Each row is one GPS observation and candidate arc pairing. The input contains trip_id, observation_id, arc_id, reference_arc_id, from_node, to_node and numeric features. The harness sets the label to 1 when arc_id equals reference_arc_id and reads the probability for class 1 by its position in classes_. Keep section IDs, timestamps used as identifiers, and any reference-derived values out of the model's feature columns. Retain candidates from observations whose reference arc is absent.

The runner performs four GroupKFold splits on training trips and then fits a fresh estimator on all training trips for the separate student-test evaluation. HistGradientBoosting uses a fixed iteration budget here. Internal early stopping is disabled because a random internal validation split would not respect trip grouping. Tune using the training folds and use the student-test partition for development comparisons. Keep evaluator trips for the demo.

### Output

| File | Contents |
|---|---|
| histgb_metrics.csv | One row per CV fold and one student_test row. Includes split, trip_ids, sample counts, candidate accuracy, macro precision/recall/F1, log loss, observation top-1, candidate miss rate, direct arc continuity and fit/inference timings. |
| comparison.csv | Report-ready summary with mean and sample SD of CV macro F1, plus student-test metrics. |
| comparison.png | Chart generated from the reported metrics. |
| run_config.json | Model parameters, feature order, fold count and dataset paths. |

Candidate classification metrics use binary labels. Observation top-1 ranks candidates within each observation. The reported continuity is a direct arc adjacency proxy on the classifier's winners. Values for accuracy, F1, miss rate and continuity are fractions on a 0–1 scale.

## Proposed model API contract

Laptop 2 sends POST /predict with Content-Type application/json.

~~~json
{"trip_id":"00000007"}
~~~

Laptop 1 returns HTTP 200 with these fields. The values below illustrate the shape.

~~~json
{
  "trip_id": "00000007",
  "predicted_route": {
    "type": "LineString",
    "coordinates": [[54.963519, 56.251559], [54.963143, 56.251523]]
  },
  "metrics": {
    "pointwise_accuracy": 0.78,
    "correct_observations": 78,
    "scored_observations": 100
  }
}
~~~

- Treat trip_id as a string and preserve leading zeros.
- Return the decoded route as a GeoJSON geometry in WGS84 longitude, latitude order. A continuous path uses LineString. Use MultiLineString for separate stretches so the UI does not draw a line across a gap.
- Compute pointwise_accuracy on the server from the final decoded predictions. Its denominator is scored_observations. The UI formats the fraction as a percentage. This differs from the harness CSV's classifier-only top-1.
- Keep reference arcs and per-point labels on the server. Return HTTP 404 with a JSON detail message for an unknown trip ID. FastAPI can return HTTP 422 for invalid request bodies.
- If the UI is served from a separate browser origin, allow its origin in FastAPI CORS settings.
