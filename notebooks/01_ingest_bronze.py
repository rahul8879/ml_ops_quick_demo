# Databricks notebook source
# MAGIC %md # Bronze ingestion

# COMMAND ----------

dbutils.widgets.text("catalog",         "wcqmlopsdemo",                                            "catalog")
dbutils.widgets.text("bronze_schema",   "bronze",                                                  "bronze schema")
dbutils.widgets.text("bronze_table",    "raw_claims",                                              "bronze table")
dbutils.widgets.text("source_path",     "/Volumes/wcqmlopsdemo/raw/claims/",                       "source path")
dbutils.widgets.text("checkpoint_path", "/Volumes/wcqmlopsdemo/raw/checkpoints/bronze_raw_claims", "checkpoint path")
dbutils.widgets.text("run_date",        "",                                                        "run date")

catalog         = dbutils.widgets.get("catalog")
bronze_schema   = dbutils.widgets.get("bronze_schema")
bronze_table    = dbutils.widgets.get("bronze_table")
source_path     = dbutils.widgets.get("source_path")
checkpoint_path = dbutils.widgets.get("checkpoint_path")
run_date        = dbutils.widgets.get("run_date")

bronze_fqn = f"{catalog}.{bronze_schema}.{bronze_table}"

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, input_file_name, lit

schema_location = f"{checkpoint_path}/_schema"

bronze_stream = (
    spark.readStream
         .format("cloudFiles")
         .option("cloudFiles.format", "csv")
         .option("cloudFiles.schemaLocation", schema_location)
         .option("cloudFiles.inferColumnTypes", "true")
         .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
         .option("header", "true")
         .option("rescuedDataColumn", "_rescued_data")
         .load(source_path)
         .withColumn("ingestion_timestamp", current_timestamp())
         .withColumn("source_file",         input_file_name())
         .withColumn("run_date",            lit(run_date))
)

(
    bronze_stream.writeStream
         .format("delta")
         .option("checkpointLocation", checkpoint_path)
         .option("mergeSchema", "true")
         .outputMode("append")
         .trigger(availableNow=True)
         .toTable(bronze_fqn)
)

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {bronze_fqn}
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact'   = 'true',
        'quality'                          = 'bronze'
    )
""")

spark.sql(f"ALTER TABLE {bronze_fqn} SET TAGS ('layer' = 'bronze', 'domain' = 'insurance_fraud')")

# COMMAND ----------

row_count = spark.table(bronze_fqn).count()
print(f"{bronze_fqn} -> {row_count:,}")
display(spark.table(bronze_fqn).limit(5))

# COMMAND ----------

dbutils.notebook.exit(f"bronze_rows={row_count}")
