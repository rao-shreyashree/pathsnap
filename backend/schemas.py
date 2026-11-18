"""Request/response models. Shape follows B's API_CONTRACT.md (UI on Laptop 2)."""
from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    trip_id: str = Field(min_length=1, examples=["00000007"])


class GpsPoint(BaseModel):
    obs_idx: int
    lat: float
    lon: float
    ts: float


class PredictionResponse(BaseModel):
    trip_id: str
    gps: List[GpsPoint]
    # one [from, to] segment per predicted arc, in order; each point is [lat, lon]
    predicted_route: List[List[List[float]]]
    # the UI shows every entry, so keep values numeric or short strings
    metrics: Dict[str, Union[float, int, str]]
    warnings: List[str] = []


class TripList(BaseModel):
    trips: List[str]


class Health(BaseModel):
    status: str
    trips_loaded: int
    hmm_enabled: bool
