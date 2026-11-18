from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    trip_id: str
    predicted_route: dict  # GeoJSON geometry
    metrics: dict[str, float | None]