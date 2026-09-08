import os
import sys

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def create_local_spark(
    app_name: str,
) -> SparkSession:
    os.environ.setdefault(
        "PYSPARK_PYTHON",
        sys.executable,
    )

    os.environ.setdefault(
        "PYSPARK_DRIVER_PYTHON",
        sys.executable,
    )

    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config(
            "spark.sql.shuffle.partitions",
            "8",
        )
        .config(
            "spark.sql.adaptive.enabled",
            "true",
        )
        .getOrCreate()
    )


def generate_telemetry_events(
    spark: SparkSession,
    n_events: int = 1_000_000,
    events_per_trip: int = 100,
) -> DataFrame:
    events = spark.range(n_events).withColumnRenamed(
        "id",
        "event_id",
    )

    events = events.withColumn(
        "trip_id",
        F.floor(F.col("event_id") / F.lit(events_per_trip)).cast("long"),
    )

    events = events.withColumn(
        "event_idx",
        F.pmod(
            F.col("event_id"),
            F.lit(events_per_trip),
        ).cast("int"),
    )

    events = events.withColumn(
        "driver_id",
        F.pmod(
            F.col("trip_id"),
            F.lit(5_000),
        ).cast("long"),
    )

    events = events.withColumn(
        "city_id",
        F.pmod(
            F.col("trip_id"),
            F.lit(20),
        ).cast("int"),
    )

    events = events.withColumn(
        "vehicle_type",
        F.when(
            F.pmod(
                F.col("trip_id"),
                F.lit(3),
            )
            == 0,
            "sedan",
        )
        .when(
            F.pmod(
                F.col("trip_id"),
                F.lit(3),
            )
            == 1,
            "suv",
        )
        .otherwise("hatchback"),
    )

    events = events.withColumn(
        "speed",
        F.round(
            25 + F.rand(seed=42) * 95,
            2,
        ),
    )

    events = events.withColumn(
        "acceleration",
        F.round(
            F.randn(seed=43) * 1.8,
            3,
        ),
    )

    events = events.withColumn(
        "_maneuver_draw",
        F.rand(seed=44),
    )

    events = events.withColumn(
        "maneuver_type",
        F.when(
            F.col("_maneuver_draw") < 0.03,
            "brake",
        )
        .when(
            F.col("_maneuver_draw") < 0.06,
            "acceleration",
        )
        .when(
            F.col("_maneuver_draw") < 0.10,
            "turn",
        )
        .otherwise("normal"),
    ).drop("_maneuver_draw")

    return events


def add_event_features(
    events: DataFrame,
) -> DataFrame:
    trip_window = Window.partitionBy("trip_id").orderBy("event_idx")

    enriched = events.withColumn(
        "previous_speed",
        F.lag("speed").over(trip_window),
    )

    enriched = enriched.withColumn(
        "delta_speed",
        F.col("speed") - F.col("previous_speed"),
    )

    enriched = enriched.withColumn(
        "harsh_brake",
        ((F.col("delta_speed") < -12) | (F.col("acceleration") < -3)).cast("int"),
    )

    enriched = enriched.withColumn(
        "harsh_acceleration",
        (F.col("acceleration") > 3).cast("int"),
    )

    return enriched


def aggregate_trip_features(
    events: DataFrame,
) -> DataFrame:
    features = events.groupBy(
        "trip_id",
        "driver_id",
        "city_id",
        "vehicle_type",
    ).agg(
        F.count("*").alias("event_count"),
        F.avg("speed").alias("speed_mean"),
        F.stddev_pop("speed").alias("speed_std"),
        F.max("speed").alias("speed_max"),
        F.stddev_pop("acceleration").alias("acceleration_std"),
        F.sum("harsh_brake").alias("harsh_braking_count"),
        F.sum("harsh_acceleration").alias("harsh_acceleration_count"),
        ((F.max("event_idx") + F.lit(1)) / F.lit(2.0)).alias("trip_duration_minutes"),
    )

    features = features.withColumn(
        "risk_label",
        (
            (F.col("speed_mean") > 72)
            | (F.col("acceleration_std") > 2.2)
            | (F.col("harsh_braking_count") >= 4)
        ).cast("int"),
    )

    return features


def create_city_dimension(
    spark: SparkSession,
) -> DataFrame:
    rows = []

    for city_id in range(20):
        if city_id % 5 == 0:
            risk_tier = "high"
        elif city_id % 3 == 0:
            risk_tier = "medium"
        else:
            risk_tier = "low"

        rows.append(
            (
                city_id,
                f"city_{city_id}",
                risk_tier,
            )
        )

    return spark.createDataFrame(
        rows,
        schema=[
            "city_id",
            "city_name",
            "city_risk_tier",
        ],
    )


def write_trip_features(
    features: DataFrame,
    output_path: str,
) -> None:
    (
        features.repartition(
            8,
            "city_id",
        )
        .write.mode("overwrite")
        .partitionBy("city_id")
        .parquet(output_path)
    )
