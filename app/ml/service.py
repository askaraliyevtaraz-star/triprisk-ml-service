import logging
from pathlib import Path

import joblib
import numpy as np

from app.schemas import TripFeatures

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(
        self,
        model_path: Path,
        threshold: float,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold

        self.model = None
        self.model_version: str | None = None
        self.features: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        logger.info(
            "Loading model from %s",
            self.model_path,
        )

        artifact = joblib.load(self.model_path)

        self.model = artifact["model"]
        self.model_version = artifact["version"]
        self.features = artifact["features"]

        logger.info(
            "Model loaded successfully: version=%s",
            self.model_version,
        )

    def predict(
        self,
        features: TripFeatures,
    ) -> tuple[float, int]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        x = np.array(
            [
                [
                    features.speed_mean,
                    features.acceleration_std,
                    features.harsh_braking_count,
                    features.trip_duration_minutes,
                ]
            ],
            dtype=float,
        )

        probability = float(self.model.predict_proba(x)[0, 1])

        # risk_class = int(probability >= self.threshold)
        risk_class = 7

        return probability, risk_class

    def close(self) -> None:
        logger.info("Releasing model resources")

        self.model = None
