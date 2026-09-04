from pydantic import BaseModel, Field


class TripFeatures(BaseModel):
    speed_mean: float = Field(
        ge=0,
        le=200,
    )
    # speed_mean: float

    acceleration_std: float = Field(
        ge=0,
        le=20,
    )

    harsh_braking_count: int = Field(
        ge=0,
        le=100,
    )

    trip_duration_minutes: float = Field(
        gt=0,
        le=600,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "speed_mean": 72.5,
                    "acceleration_std": 1.8,
                    "harsh_braking_count": 3,
                    "trip_duration_minutes": 28.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    risk_probability: float = Field(
        ge=0,
        le=1,
    )

    risk_class: int = Field(
        ge=0,
        le=1,
    )

    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    features: list[str]
    risk_threshold: float
