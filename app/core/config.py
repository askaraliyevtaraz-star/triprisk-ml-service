from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TripRisk API"
    app_version: str = "0.3.0"

    model_path: Path = Path("artifacts/model.joblib")
    risk_threshold: float = 0.5

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="TRIPRISK_",
        env_file=".env",
        extra="ignore",
    )

    model_source: Literal[
        "local",
        "mlflow",
    ] = "local"

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"

    registered_model_name: str = "TripRiskClassifier"

    model_alias: str = "champion"


@lru_cache
def get_settings() -> Settings:
    return Settings()
