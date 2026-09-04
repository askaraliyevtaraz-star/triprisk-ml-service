from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
N_SAMPLES = 5000

rng = np.random.default_rng(RANDOM_SEED)


# Synthetic trip features
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


X = np.column_stack(
    [
        speed_mean,
        acceleration_std,
        harsh_braking_count,
        trip_duration_minutes,
    ]
)


# Synthetic underlying risk function
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


model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)

model.fit(X, y)


artifact = {
    "model": model,
    "version": "1.0.0",
    "features": [
        "speed_mean",
        "acceleration_std",
        "harsh_braking_count",
        "trip_duration_minutes",
    ],
}


output_path = Path("artifacts/model.joblib")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    artifact,
    output_path,
)


print(f"Model saved to: {output_path}")
print(f"Positive rate: {y.mean():.3f}")
