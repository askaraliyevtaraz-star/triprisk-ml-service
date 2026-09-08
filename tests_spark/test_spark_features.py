import pytest
from pyspark.sql import SparkSession

from pipelines.spark_features import (
    add_event_features,
    aggregate_trip_features,
    generate_telemetry_events,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("TripRisk-Spark-Tests")
        .config(
            "spark.sql.shuffle.partitions",
            "2",
        )
        .getOrCreate()
    )

    yield session

    session.stop()


def test_event_generation(
    spark: SparkSession,
) -> None:
    events = generate_telemetry_events(
        spark,
        n_events=1_000,
        events_per_trip=100,
    )

    assert events.count() == 1_000


def test_trip_aggregation(
    spark: SparkSession,
) -> None:
    events = generate_telemetry_events(
        spark,
        n_events=1_000,
        events_per_trip=100,
    )

    enriched = add_event_features(events)

    features = aggregate_trip_features(enriched)

    assert features.count() == 10


def test_required_features_exist(
    spark: SparkSession,
) -> None:
    events = generate_telemetry_events(
        spark,
        n_events=1_000,
        events_per_trip=100,
    )

    features = aggregate_trip_features(add_event_features(events))

    required = {
        "speed_mean",
        "acceleration_std",
        "harsh_braking_count",
        "trip_duration_minutes",
        "risk_label",
    }

    assert required.issubset(set(features.columns))
