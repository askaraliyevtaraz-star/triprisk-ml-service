import argparse

from pyspark.sql import functions as F

from pipelines.spark_features import (
    add_event_features,
    aggregate_trip_features,
    create_city_dimension,
    create_local_spark,
    generate_telemetry_events,
    write_trip_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--events",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=("airflow/data/spark_trip_features"),
    )

    args = parser.parse_args()

    spark = create_local_spark("TripRisk-Feature-Pipeline")

    try:
        events = generate_telemetry_events(
            spark,
            n_events=args.events,
        )

        events = add_event_features(events)

        features = aggregate_trip_features(events)

        city_dim = create_city_dimension(spark)

        features = features.join(
            F.broadcast(city_dim),
            on="city_id",
            how="left",
        )

        print(
            "Feature rows:",
            features.count(),
        )

        features.printSchema()

        write_trip_features(
            features,
            args.output,
        )

        print(
            "Features written to:",
            args.output,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
