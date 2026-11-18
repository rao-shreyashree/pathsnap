import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
pipe = Pipeline()


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "trips_loaded": len(pipe.trip_ids())}


@app.get("/trips")
def trips():
    return {"trips": pipe.trip_ids()}


@app.post("/predict")
def predict(req: PredictionRequest):
    if req.trip_id not in pipe.tracks:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown trip_id")
    return pipe.predict(req.trip_id)