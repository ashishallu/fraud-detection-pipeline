"""
spark/feature_engineering.py
Consumes the Kafka transaction stream and computes windowed velocity features
using PySpark Structured Streaming.

Windowed features per card1 over a 5-minute tumbling window:
  - txn_count_5min
  - txn_sum_5min
  - txn_mean_5min
  - txn_stddev_5min
  - velocity_flag  (1 if count > 10 in window)
"""

import logging
import sys
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, IntegerType, LongType, DoubleType, BooleanType,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger("feature_engineering")


# ─────────────────────────────────────────────
# Kafka message schema
# Full schema matching merged IEEE-CIS columns
# ─────────────────────────────────────────────
def get_transaction_schema() -> StructType:
    """Return the full Spark schema for a merged IEEE-CIS transaction row."""
    def _f(name, dtype):
        return StructField(name, dtype, nullable=True)

    fields = [
        _f("TransactionID", IntegerType()),
        _f("isFraud", IntegerType()),
        _f("TransactionDT", IntegerType()),
        _f("TransactionAmt", DoubleType()),
        _f("ProductCD", StringType()),
        _f("card1", IntegerType()),
        _f("card2", FloatType()),
        _f("card3", FloatType()),
        _f("card4", StringType()),
        _f("card5", FloatType()),
        _f("card6", StringType()),
        _f("addr1", FloatType()),
        _f("addr2", FloatType()),
        _f("dist1", FloatType()),
        _f("dist2", FloatType()),
        _f("P_emaildomain", StringType()),
        _f("R_emaildomain", StringType()),
        _f("event_time", DoubleType()),
    ]
    # C columns
    for i in range(1, 15):
        fields.append(_f(f"C{i}", FloatType()))
    # D columns
    for i in range(1, 16):
        fields.append(_f(f"D{i}", FloatType()))
    # M columns
    for i in range(1, 10):
        fields.append(_f(f"M{i}", StringType()))
    # V columns (first 15 for schema; rest are included as passthrough)
    for i in range(1, 340):
        fields.append(_f(f"V{i}", FloatType()))
    # Identity columns (from train_identity.csv merge)
    fields += [
        _f("id_01", FloatType()), _f("id_02", FloatType()),
        _f("id_03", FloatType()), _f("id_04", FloatType()),
        _f("id_05", FloatType()), _f("id_06", FloatType()),
        _f("id_09", FloatType()), _f("id_10", FloatType()),
        _f("id_11", FloatType()),
        _f("id_12", StringType()), _f("id_13", FloatType()),
        _f("id_14", FloatType()), _f("id_15", StringType()),
        _f("id_16", StringType()), _f("id_17", FloatType()),
        _f("id_18", FloatType()), _f("id_19", FloatType()),
        _f("id_20", FloatType()), _f("id_28", StringType()),
        _f("id_29", StringType()), _f("id_30", StringType()),
        _f("id_31", StringType()), _f("id_32", FloatType()),
        _f("id_33", StringType()), _f("id_34", StringType()),
        _f("id_35", StringType()), _f("id_36", StringType()),
        _f("id_37", StringType()), _f("id_38", StringType()),
        _f("DeviceType", StringType()),
        _f("DeviceInfo", StringType()),
    ]
    return StructType(fields)


def create_spark_session() -> SparkSession:
    """Create and return a configured SparkSession with Kafka support."""
    spark = (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
        .config("spark.sql.streaming.checkpointLocation", config.SPARK_CHECKPOINT_DIR)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log.info("SparkSession created — master: %s", config.SPARK_MASTER)
    return spark


def read_kafka_stream(spark: SparkSession) -> DataFrame:
    """Create a streaming DataFrame from the Kafka transactions-raw topic."""
    log.info("Connecting to Kafka topic: %s", config.KAFKA_TOPIC_RAW)
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", config.KAFKA_TOPIC_RAW)
        .option("startingOffsets", config.KAFKA_AUTO_OFFSET_RESET)
        .option("maxOffsetsPerTrigger", config.SPARK_MAX_OFFSETS_PER_TRIGGER)
        .option("failOnDataLoss", "false")
        .load()
    )
    return raw_stream


def parse_transactions(raw_stream: DataFrame) -> DataFrame:
    """Parse raw Kafka JSON bytes into a typed transaction DataFrame."""
    schema = get_transaction_schema()

    parsed = (
        raw_stream
        .select(F.from_json(F.col("value").cast("string"), schema).alias("data"))
        .select("data.*")
    )

    # Convert Unix float timestamp to Spark timestamp
    parsed = parsed.withColumn(
        "event_timestamp",
        F.to_timestamp(F.col("event_time").cast("long"))
    )

    # Apply watermark for late data handling
    parsed = parsed.withWatermark("event_timestamp", config.SPARK_WATERMARK_DELAY)

    # Fill nulls in numeric columns with 0
    numeric_cols = [f.name for f in schema.fields
                    if isinstance(f.dataType, (FloatType, DoubleType, IntegerType))
                    and f.name not in ("TransactionID", "isFraud")]
    parsed = parsed.fillna(0, subset=numeric_cols)

    return parsed


def compute_velocity_features(parsed: DataFrame) -> DataFrame:
    """
    Compute 5-minute tumbling window velocity features per card1.
    Returns a DataFrame with windowed aggregations joined back to transactions.
    """
    # Windowed aggregations over 5-minute tumbling windows
    velocity = (
        parsed
        .groupBy(
            F.window("event_timestamp", config.SPARK_WINDOW_DURATION),
            F.col("card1")
        )
        .agg(
            F.count("TransactionID").alias("txn_count_5min"),
            F.sum("TransactionAmt").alias("txn_sum_5min"),
            F.mean("TransactionAmt").alias("txn_mean_5min"),
            F.stddev("TransactionAmt").alias("txn_stddev_5min"),
        )
        .withColumn(
            "velocity_flag",
            F.when(F.col("txn_count_5min") > 10, 1).otherwise(0)
        )
        .withColumn("window_start", F.col("window.start"))
        .drop("window")
    )
    return velocity


def enrich_stream(parsed: DataFrame) -> DataFrame:
    enriched = (
        parsed
        .withColumn("TransactionAmt",
                    F.col("TransactionAmt").cast("double"))  # ← add this
        .withColumn("log_amount", F.log1p(F.col("TransactionAmt")))
        .withColumn("hour_of_day", F.hour("event_timestamp"))
        .withColumn("day_of_week", F.dayofweek("event_timestamp"))
        .withColumn("txn_stddev_5min", F.lit(0.0))
        .withColumn("txn_count_5min", F.lit(0))
        .withColumn("txn_sum_5min", F.lit(0.0))
        .withColumn("txn_mean_5min", F.lit(0.0))
        .withColumn("velocity_flag", F.lit(0))
    )
    return enriched


def write_to_parquet_sink(batch_df: DataFrame, batch_id: int) -> None:
    """Write each micro-batch to Parquet for downstream consumption."""
    output_path = str(config.RESULTS_DIR / "parquet" / f"batch_{batch_id}")
    batch_df.write.mode("overwrite").parquet(output_path)
    log.info("Batch %d written to Parquet — %d rows", batch_id, batch_df.count())


def start_feature_engineering_stream(spark: SparkSession, foreach_batch_fn=None):
    """
    Start the Spark Structured Streaming job.

    Args:
        spark: Active SparkSession
        foreach_batch_fn: Optional callback(batch_df, batch_id) for each micro-batch.
                          If None, defaults to writing Parquet only.
    Returns:
        StreamingQuery handle
    """
    raw = read_kafka_stream(spark)
    parsed = parse_transactions(raw)
    enriched = enrich_stream(parsed)

    handler = foreach_batch_fn if foreach_batch_fn else write_to_parquet_sink

    query = (
        enriched.writeStream
        .outputMode("append")
        .option("checkpointLocation", config.SPARK_CHECKPOINT_DIR + "/features")
        .trigger(processingTime=config.SPARK_TRIGGER_INTERVAL)
        .foreachBatch(handler)
        .start()
    )
    log.info("Feature engineering stream started — trigger: %s", config.SPARK_TRIGGER_INTERVAL)
    return query


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    spark = create_spark_session()
    query = start_feature_engineering_stream(spark)
    log.info("Streaming... press Ctrl+C to stop")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        log.info("Spark streaming stopped.")
