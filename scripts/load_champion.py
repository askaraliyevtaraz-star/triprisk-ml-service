import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "TripRiskClassifier"

MODEL_ALIAS = "champion"


mlflow.set_tracking_uri(TRACKING_URI)


client = MlflowClient()


model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS,
)


print(
    "Champion version:",
    model_version.version,
)


model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


model = mlflow.sklearn.load_model(model_uri)


x = np.array(
    [
        [
            80.0,
            2.5,
            4,
            30.0,
        ]
    ]
)


probability = model.predict_proba(x)[0, 1]


print(
    "Risk probability:",
    float(probability),
)
