from fastapi.testclient import TestClient


def test_predict_valid_request(
    client: TestClient,
) -> None:
    payload = {
        "speed_mean": 80,
        "acceleration_std": 2.5,
        "harsh_braking_count": 4,
        "trip_duration_minutes": 30,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["risk_probability"] <= 1

    assert data["risk_class"] in (0, 1)
    # assert data["risk_class"] == 7

    assert data["model_version"] == "1.0.0"


def test_predict_rejects_missing_feature(
    client: TestClient,
) -> None:
    payload = {
        "speed_mean": 80,
        "acceleration_std": 2.5,
        "harsh_braking_count": 4,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_predict_rejects_invalid_type(
    client: TestClient,
) -> None:
    payload = {
        "speed_mean": "Radiohead",
        "acceleration_std": 2.5,
        "harsh_braking_count": 4,
        "trip_duration_minutes": 30,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_risky_trip_scores_higher(
    client: TestClient,
) -> None:
    safe_payload = {
        "speed_mean": 35,
        "acceleration_std": 0.3,
        "harsh_braking_count": 0,
        "trip_duration_minutes": 15,
    }

    risky_payload = {
        "speed_mean": 120,
        "acceleration_std": 6,
        "harsh_braking_count": 15,
        "trip_duration_minutes": 90,
    }

    safe_response = client.post(
        "/predict",
        json=safe_payload,
    )

    risky_response = client.post(
        "/predict",
        json=risky_payload,
    )

    safe_probability = safe_response.json()["risk_probability"]

    risky_probability = risky_response.json()["risk_probability"]

    assert risky_probability > safe_probability
