# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze Ingestion (Vehicle Insurance Claims)
# MAGIC
# MAGIC **Purpose:** Incrementally ingest raw Vehicle Insurance Claim Fraud CSV files from a
# MAGIC Unity Catalog Volume into a Delta Bronze table using Databricks Auto Loader
# MAGIC (`cloudFiles`). Bronze = raw, append-only, schema-on-read + rescued data.
# MAGIC
# MAGIC **Inputs**
# MAGIC - Volume path: `/Volumes/insurance_demo/raw/claims/` (CSV files)
# MAGIC
# MAGIC **Outputs**
# MAGIC - Delta table: `insurance_demo.bronze.raw_claims`
# MAGIC - Checkpoint location: `/Volumes/insurance_demo/raw/_checkpoints/bronze_raw_claims`
# MAGIC
# MAGIC **Runtime:** Databricks Runtime 15.4 LTS ML

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets / configurable parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "insurance_demo", "Unity Catalog")
dbutils.widgets.text("bronze_schema", "bronze", "Bronze schema")
dbutils.widgets.text("bronze_table", "raw_claims", "Bronze table")
dbutils.widgets.text(
    "source_path",
    "/Volumes/insurance_demo/raw/claims/",
    "Source volume path"
)
dbutils.widgets.text(
    "checkpoint_path",
    "/Volumes/insurance_demo/raw/_checkpoints/bronze_raw_claims",
    "Auto Loader checkpoint"
)
dbutils.widgets.text("run_date", "", "Run date (YYYY-MM-DD, optional)")

catalog          = dbutils.widgets.get("catalog")
bronze_schema    = dbutils.widgets.get("bronze_schema")
bronze_table     = dbutils.widgets.get("bronze_table")
source_path      = dbutils.widgets.get("source_path")
checkpoint_path  = dbutils.widgets.get("checkpoint_path")
run_date         = dbutils.widgets.get("run_date")

bronze_fqn = f"{catalog}.{bronze_schema}.{bronze_table}"
print(f"Target bronze table: {bronze_fqn}")
print(f"Source path:         {source_path}")
print(f"Checkpoint path:     {checkpoint_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ensure catalog / schema exist

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {catalog}.{bronze_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Auto Loader (cloudFiles) → Bronze Delta
# MAGIC
# MAGIC We use `cloudFiles` in `availableNow=True` trigger mode so the notebook can run
# MAGIC as a scheduled batch job (Databricks Workflow) while still using the streaming
# MAGIC semantics of Auto Loader (incremental file discovery, exactly-once processing,
# MAGIC schema evolution).

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, input_file_name, lit

# Schema location lives next to the checkpoint so inferred schema is preserved
# across runs and schema evolution is supported.
schema_location = f"{checkpoint_path}/_schema"

bronze_stream = (
    spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", schema_location)
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("header", "true")
        # rescued data column captures values that don't match inferred schema
        .option("rescuedDataColumn", "_rescued_data")
        .load(source_path)
        .withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_file",         input_file_name())
        .withColumn("run_date",            lit(run_date))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write stream to bronze Delta table (append, trigger availableNow)

# COMMAND ----------

(
    bronze_stream.writeStream
        .format("delta")
        .option("checkpointLocation", checkpoint_path)
        .option("mergeSchema", "true")
        .outputMode("append")
        .trigger(availableNow=True)          # batch-style run inside a job
        .toTable(bronze_fqn)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Set table properties, comments, and tags on bronze

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {bronze_fqn}
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact'   = 'true',
        'quality'                          = 'bronze',
        'pipelines.autoOptimize.managed'   = 'true'
    )
""")

spark.sql(f"""
    COMMENT ON TABLE {bronze_fqn} IS
    'Raw, append-only ingestion of Vehicle Insurance Claim Fraud CSVs via Auto Loader.'
""")

# Unity Catalog tags — useful for governance & cost attribution
spark.sql(f"ALTER TABLE {bronze_fqn} SET TAGS ('layer' = 'bronze', 'domain' = 'insurance_fraud')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Quick sanity check

# COMMAND ----------

row_count = spark.table(bronze_fqn).count()
print(f"[bronze] {bronze_fqn} row count = {row_count:,}")

display(spark.table(bronze_fqn).limit(5))

# COMMAND ----------

dbutils.notebook.exit(f"bronze_rows={row_count}")
