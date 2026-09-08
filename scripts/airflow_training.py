import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from scripts.train_mlflow import (
    EXPERIMENT_NAME,
    MODEL_ALIAS,
    RANDOM_SEED,
    REGISTERED_MODEL_NAME,
    build_candidates,
    evaluate_model,
    generate_dataset,
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000",
)


def prepare_dataset(
    output_path: str,
) -> str:
    x, y = generate_dataset()

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        x=x,
        y=y,
    )

    return str(path)


def train_candidate(
    data_path: str,
    candidate_name: str,
) -> dict[str, object]:
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    data = np.load(data_path)

    x = data["x"]
    y = data["y"]

    (
        x_train,
        x_valid,
        y_train,
        y_valid,
    ) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    candidates = {name: (model, params) for name, model, params in build_candidates()}

    if candidate_name not in candidates:
        raise ValueError(f"Unknown candidate: {candidate_name}")

    model, params = candidates[candidate_name]

    with mlflow.start_run(run_name=(f"airflow-{candidate_name}")) as run:
        mlflow.log_params(params)

        mlflow.set_tag(
            "orchestrator",
            "airflow",
        )

        mlflow.set_tag(
            "candidate",
            candidate_name,
        )

        model.fit(
            x_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            x_valid,
            y_valid,
        )

        mlflow.log_metrics(metrics)

        predictions = model.predict(x_valid)

        signature = infer_signature(
            x_valid,
            predictions,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_valid[:5],
        )

        return {
            "candidate": candidate_name,
            "run_id": run.info.run_id,
            "pr_auc": float(metrics["pr_auc"]),
            "roc_auc": float(metrics["roc_auc"]),
            "model_uri": (model_info.model_uri),
        }


def select_best(
    results: list[dict[str, object]],
) -> dict[str, object]:
    if not results:
        raise ValueError("No model results provided")

    best = max(
        results,
        key=lambda result: float(result["pr_auc"]),
    )

    return best


def register_champion(
    best: dict[str, object],
) -> dict[str, str]:
    mlflow.set_tracking_uri(TRACKING_URI)

    registered = mlflow.register_model(
        model_uri=str(best["model_uri"]),
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
        registered.version,
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        registered.version,
        "orchestrator",
        "airflow",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        registered.version,
        "selection_metric",
        "pr_auc",
    )

    return {
        "model_name": (REGISTERED_MODEL_NAME),
        "version": str(registered.version),
        "alias": MODEL_ALIAS,
        "candidate": str(best["candidate"]),
    }
