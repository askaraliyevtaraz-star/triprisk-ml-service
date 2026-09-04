from fastapi.testclient import TestClient


def test_health(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info(
    client: TestClient,
) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_version"] == "1.0.0"

    assert data["features"] == [
        "speed_mean",
        "acceleration_std",
        "harsh_braking_count",
        "trip_duration_minutes",
    ]

    assert 0 <= data["risk_threshold"] <= 1
