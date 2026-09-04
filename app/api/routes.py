from fastapi import APIRouter, Request

from app.ml.service import ModelService
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TripFeatures,
)

router = APIRouter()


def get_model_service(
    request: Request,
) -> ModelService:
    return request.app.state.model_service


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "TripRisk API is running"}


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health(
    request: Request,
) -> HealthResponse:
    service = get_model_service(request)

    return HealthResponse(
        status=("ok" if service.is_loaded else "error"),
        model_loaded=service.is_loaded,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info(
    request: Request,
) -> ModelInfoResponse:
    service = get_model_service(request)

    return ModelInfoResponse(
        model_version=service.model_version or "unknown",
        features=service.features,
        risk_threshold=service.threshold,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    features: TripFeatures,
    request: Request,
) -> PredictionResponse:
    service = get_model_service(request)

    probability, risk_class = service.predict(features)

    return PredictionResponse(
        risk_probability=probability,
        risk_class=risk_class,
        model_version=service.model_version or "unknown",
    )
