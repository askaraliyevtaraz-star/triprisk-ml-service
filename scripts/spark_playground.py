import time

from pyspark.sql import functions as F

from pipelines.spark_features import (
    add_event_features,
    aggregate_trip_features,
    create_city_dimension,
    create_local_spark,
    generate_telemetry_events,
)

spark = create_local_spark("TripRisk-Day7")


print(
    "Spark UI:",
    spark.sparkContext.uiWebUrl,
)


events = generate_telemetry_events(
    spark,
    n_events=1_000_000,
)


print("Schema:")

events.printSchema()


print(
    "Partitions:",
    events.rdd.getNumPartitions(),
)


enriched = add_event_features(events)


features = aggregate_trip_features(enriched)


print("We have defined transformations.")

print("No explicit action has been called on features yet.")


start = time.perf_counter()

row_count = features.count()

elapsed = time.perf_counter() - start


print(f"Trips: {row_count:,}")

print(f"First count: {elapsed:.3f}s")


features.show(
    10,
    truncate=False,
)


print("\nExecution plan:")

features.explain(mode="formatted")

features.createOrReplaceTempView("trip_features")


spark.sql(
    """
    SELECT
        vehicle_type,
        COUNT(*) AS trips,
        AVG(speed_mean) AS avg_speed,
        AVG(acceleration_std)
            AS avg_acceleration_std,
        AVG(risk_label)
            AS risk_rate
    FROM trip_features
    GROUP BY vehicle_type
    ORDER BY risk_rate DESC
    """
).show(truncate=False)

print(features.rdd.getNumPartitions())

features_8 = features.repartition(
    8,
    "city_id",
)

print(features_8.rdd.getNumPartitions())

city_dim = create_city_dimension(spark)


joined = features.join(
    F.broadcast(city_dim),
    on="city_id",
    how="left",
)


joined.show(
    10,
    truncate=False,
)

joined.explain(mode="formatted")

joined.cache()
# cache()

start = time.perf_counter()

joined.count()

print(
    "Cache materialization:",
    time.perf_counter() - start,
)

start = time.perf_counter()

joined.count()

print(
    "Cached count:",
    time.perf_counter() - start,
)


input("\nOpen Spark UI, inspect it, then press Enter...")

joined.unpersist()

spark.stop()
