import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from app.schemas import TripFeatures

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(
        self,
        model_path: Path,
        threshold: float,
        model_source: str = "local",
        mlflow_tracking_uri: str = ("http://127.0.0.1:5000"),
        registered_model_name: str = ("TripRiskClassifier"),
        model_alias: str = "champion",
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold

        self.model_source = model_source

        self.mlflow_tracking_uri = mlflow_tracking_uri

        self.registered_model_name = registered_model_name

        self.model_alias = model_alias

        self.model = None

        self.model_version: str | None = None

        self.features: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model_source == "local":
            self._load_local()

        elif self.model_source == "mlflow":
            self._load_mlflow()

        else:
            raise ValueError(f"Unknown model source: {self.model_source}")

    def _load_local(self) -> None:
        logger.info(
            "Loading local model from %s",
            self.model_path,
        )

        artifact = joblib.load(self.model_path)

        self.model = artifact["model"]

        self.model_version = artifact["version"]

        self.features = artifact["features"]

        logger.info(
            "Local model loaded: version=%s",
            self.model_version,
        )

    def _load_mlflow(self) -> None:
        logger.info(
            "Loading model from MLflow: %s@%s",
            self.registered_model_name,
            self.model_alias,
        )

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        model_uri = f"models:/{self.registered_model_name}@{self.model_alias}"

        self.model = mlflow.sklearn.load_model(model_uri)

        client = MlflowClient()

        model_version = client.get_model_version_by_alias(
            self.registered_model_name,
            self.model_alias,
        )

        self.model_version = str(model_version.version)

        self.features = [
            "speed_mean",
            "acceleration_std",
            "harsh_braking_count",
            "trip_duration_minutes",
        ]

        logger.info(
            "MLflow model loaded: version=%s alias=%s",
            self.model_version,
            self.model_alias,
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

        risk_class = int(probability >= self.threshold)

        return probability, risk_class

    def close(self) -> None:
        logger.info("Releasing model resources")

        self.model = None
