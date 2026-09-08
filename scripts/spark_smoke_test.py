import os
import sys

from pyspark.sql import SparkSession

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


spark = (
    SparkSession.builder.master("local[*]")
    .appName("TripRisk-Day7-Smoke")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)


print("Spark version:", spark.version)
print("Spark UI:", spark.sparkContext.uiWebUrl)


df = spark.range(10)

df.show()


spark.stop()
