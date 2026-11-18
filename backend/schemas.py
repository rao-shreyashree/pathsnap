"""Request/response models. Shape agreed with B's UI (see B_TO_C_HANDOFF.md)."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    trip_id: str = Field(min_length=1, examples=["00000007"])


class RouteGeometry(BaseModel):
    """GeoJSON geometry, WGS84, coordinates are [longitude, latitude]."""
    type: Literal["LineString", "MultiLineString"]
    coordinates: List[Any]


class Metrics(BaseModel):
    pointwise_accuracy: float = Field(ge=0, le=1)   # fraction 0-1 of the FINAL (decoded) route
    correct_observations: int
    scored_observations: int


class Details(BaseModel):
    """Extra info for the report / viva. The UI may ignore it."""
    method: str                                      # "hmm_viterbi" or "classifier_only"
    classifier_only_accuracy: Optional[float] = None
    hmm_accuracy: Optional[float] = None
    n_observations: int
    n_candidate_rows: int
    elapsed_seconds: float
    warnings: List[str] = []


class PredictionResponse(BaseModel):
    trip_id: str
    predicted_route: RouteGeometry
    metrics: Metrics
    details: Optional[Details] = None


class TripList(BaseModel):
    trips: List[str]


class Health(BaseModel):
    status: str
    trips_loaded: int
    hmm_enabled: bool
    extra: Dict[str, Any] = {}
