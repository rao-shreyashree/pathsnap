"""PathSnap API (Laptop 1).  Run from the repo root:
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from backend import config as cfg
from backend.pipeline import Pipeline
from backend.schemas import Health, PredictionRequest, PredictionResponse, TripList

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("pathsnap")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipe = Pipeline()          # fails loudly at startup if model/data are missing
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("%s %s -> %d (%.0f ms) from %s", request.method, request.url.path,
             response.status_code, 1000 * (time.time() - t0),
             request.client.host if request.client else "?")
    return response


@app.get("/health", response_model=Health)
def health():
    pipe = app.state.pipe
    return Health(status="ok", trips_loaded=len(pipe.trip_ids()), hmm_enabled=cfg.USE_HMM)


@app.get("/trips", response_model=TripList)
def trips():
    return TripList(trips=app.state.pipe.trip_ids())


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    pipe = app.state.pipe
    if req.trip_id not in pipe.tracks:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown trip_id '{req.trip_id}'")
    try:
        return pipe.predict(req.trip_id)
    except Exception as exc:
        log.exception("prediction failed for %s", req.trip_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Prediction failed: {exc}")
