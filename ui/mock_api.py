"""MOCK API for testing the UI before the real API is ready (NOT part of the final system).
    pip install fastapi uvicorn
    uvicorn ui.mock_api:app --host 0.0.0.0 --port 8000"""
import math
import random
from fastapi import FastAPI, HTTPException

app = FastAPI()
TRIPS = [f"eval_trip_{i:02d}" for i in range(1, 11)]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return {"trips": TRIPS}


@app.post("/predict")
def predict(body: dict):
    tid = body.get("trip_id")
    if tid not in TRIPS:
        raise HTTPException(404, f"unknown trip {tid}")
    rnd = random.Random(tid)
    lat0, lon0 = 50.05 + rnd.random() * 0.02, 8.58 + rnd.random() * 0.02
    line = [(lat0 + i * 0.0003, lon0 + 0.004 * math.sin(i / 15)) for i in range(120)]
    gps = [{"obs_idx": i, "lat": la + rnd.gauss(0, 0.0001), "lon": lo + rnd.gauss(0, 0.0001), "ts": i}
           for i, (la, lo) in enumerate(line)]
    segs = [[list(line[i]), list(line[i + 1])] for i in range(len(line) - 1)]
    return {"trip_id": tid, "gps": gps, "predicted_route": segs,
            "metrics": {"n_obs": len(gps), "top1_classifier_only": round(rnd.uniform(.7, .85), 4),
                        "top1_with_hmm": round(rnd.uniform(.85, .97), 4),
                        "route_continuity": round(rnd.uniform(.9, 1), 4),
                        "inference_ms": round(rnd.uniform(200, 900), 1)}}
