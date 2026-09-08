from pipelines.spark_features import (
    create_local_spark,
)

spark = create_local_spark("Read-TripRisk-Features")


df = spark.read.parquet("airflow/data/spark_trip_features")


df.printSchema()

print(
    "Rows:",
    df.count(),
)


df.show(
    10,
    truncate=False,
)


spark.stop()
