import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
N_SAMPLES = 10_000

TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "TripRisk-Model-Comparison"

REGISTERED_MODEL_NAME = "TripRiskClassifier"

MODEL_ALIAS = "champion"


FEATURE_NAMES = [
    "speed_mean",
    "acceleration_std",
    "harsh_braking_count",
    "trip_duration_minutes",
]


def generate_dataset() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(RANDOM_SEED)

    speed_mean = rng.normal(
        loc=55,
        scale=18,
        size=N_SAMPLES,
    ).clip(5, 140)

    acceleration_std = rng.gamma(
        shape=2,
        scale=0.7,
        size=N_SAMPLES,
    ).clip(0, 8)

    harsh_braking_count = rng.poisson(
        lam=2,
        size=N_SAMPLES,
    ).clip(0, 20)

    trip_duration_minutes = rng.gamma(
        shape=3,
        scale=10,
        size=N_SAMPLES,
    ).clip(2, 180)

    x = np.column_stack(
        [
            speed_mean,
            acceleration_std,
            harsh_braking_count,
            trip_duration_minutes,
        ]
    )

    logits = (
        -5.0
        + 0.025 * speed_mean
        + 0.9 * acceleration_std
        + 0.25 * harsh_braking_count
        + 0.005 * trip_duration_minutes
    )

    probabilities = 1 / (1 + np.exp(-logits))

    y = rng.binomial(
        n=1,
        p=probabilities,
    )

    return x, y


def build_candidates():
    return [
        (
            "logistic-regression",
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {
                "model_family": "logistic_regression",
                "class_weight": "balanced",
                "max_iter": 1000,
            },
        ),
        (
            "random-forest",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            {
                "model_family": "random_forest",
                "n_estimators": 250,
                "max_depth": 8,
                "min_samples_leaf": 5,
            },
        ),
        (
            "hist-gradient-boosting",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                random_state=RANDOM_SEED,
            ),
            {
                "model_family": ("hist_gradient_boosting"),
                "max_iter": 150,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
            },
        ),
    ]


def evaluate_model(
    model,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> dict[str, float]:
    probabilities = model.predict_proba(x_valid)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_valid,
            probabilities,
        ),
        "pr_auc": (
            average_precision_score(
                y_valid,
                probabilities,
            )
        ),
        "log_loss": log_loss(
            y_valid,
            probabilities,
        ),
        "f1": f1_score(
            y_valid,
            predictions,
        ),
    }


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    x, y = generate_dataset()

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

    results = []

    for (
        run_name,
        model,
        params,
    ) in build_candidates():
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)

            mlflow.log_param(
                "train_rows",
                len(x_train),
            )

            mlflow.log_param(
                "validation_rows",
                len(x_valid),
            )

            mlflow.set_tag(
                "task",
                "binary_classification",
            )

            mlflow.set_tag(
                "selection_metric",
                "pr_auc",
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

            input_example = x_valid[:5]

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
            )

            with tempfile.TemporaryDirectory() as tmp:
                summary_path = Path(tmp) / "run_summary.txt"

                summary_path.write_text(
                    "\n".join(
                        [
                            f"run_name={run_name}",
                            f"run_id={run.info.run_id}",
                            f"features={FEATURE_NAMES}",
                            f"roc_auc={metrics['roc_auc']:.6f}",
                            f"pr_auc={metrics['pr_auc']:.6f}",
                            f"f1={metrics['f1']:.6f}",
                        ]
                    ),
                    encoding="utf-8",
                )

                mlflow.log_artifact(
                    str(summary_path),
                    artifact_path="reports",
                )

            results.append(
                {
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "model_uri": (model_info.model_uri),
                }
            )

            print(f"{run_name}: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    best = max(
        results,
        key=lambda item: item["pr_auc"],
    )

    print()
    print("Best model:")
    print(best)

    model_version = mlflow.register_model(
        model_uri=best["model_uri"],
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
        model_version.version,
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version.version,
        "validation_status",
        "passed",
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version.version,
        "selection_metric",
        "pr_auc",
    )

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "task",
        "trip_risk_classification",
    )

    print()
    print(f"Registered model: {REGISTERED_MODEL_NAME}")

    print(f"Version: {model_version.version}")

    print(f"Alias: @{MODEL_ALIAS}")


if __name__ == "__main__":
    main()
