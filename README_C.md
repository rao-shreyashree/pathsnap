# C: boosting model, Bayesian optimisation, FastAPI backend

## Run order (from the repo root)
    pip install -r requirements.txt -r requirements-c.txt
    python -m src.build_dataset                  # A's parquet tables
    python -m models.train_boosting              # untuned baseline via B's harness
    python -m models.tune_boosting --trials 25   # Optuna search + tuned refit + student-test eval

## API
    set PATHSNAP_MODEL=models/artifacts/boosting_tuned.joblib
    set PATHSNAP_TRACK_DIR=<folder with the .track files>
    set PATHSNAP_PARTITION=student_test          (evaluator_test on demo day)
    set PATHSNAP_USE_HMM=0                       (1 once D's decoder is merged)
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
    python scripts/smoke_test.py

Endpoints: GET /health, GET /trips, POST /predict {"trip_id": "00000007"}
Ground-truth labels are loaded only after prediction, from a second copy of the trip,
and only counts leave the server (never labels or reference arcs).
