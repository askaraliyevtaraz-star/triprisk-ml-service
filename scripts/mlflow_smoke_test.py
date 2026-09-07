import mlflow

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "TripRisk-Smoke-Test"


mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


with mlflow.start_run(run_name="hello-mlflow") as run:
    mlflow.log_param(
        "model_type",
        "dummy",
    )

    mlflow.log_metric(
        "dummy_score",
        0.42,
    )

    mlflow.set_tag(
        "purpose",
        "learning MLflow",
    )

    print(f"Run ID: {run.info.run_id}")
