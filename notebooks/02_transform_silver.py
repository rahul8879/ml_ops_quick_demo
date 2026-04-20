# Databricks notebook source
# MAGIC %md # Silver transform

# COMMAND ----------

dbutils.widgets.text("catalog",       "wcqmlopsdemo", "catalog")
dbutils.widgets.text("bronze_schema", "bronze",       "bronze schema")
dbutils.widgets.text("bronze_table",  "raw_claims",   "bronze table")
dbutils.widgets.text("silver_schema", "silver",       "silver schema")
dbutils.widgets.text("silver_table",  "clean_claims", "silver table")
dbutils.widgets.text("run_date",      "",             "run date")

catalog       = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
bronze_table  = dbutils.widgets.get("bronze_table")
silver_schema = dbutils.widgets.get("silver_schema")
silver_table  = dbutils.widgets.get("silver_table")
run_date      = dbutils.widgets.get("run_date")

bronze_fqn = f"{catalog}.{bronze_schema}.{bronze_table}"
silver_fqn = f"{catalog}.{silver_schema}.{silver_table}"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.window import Window

df = spark.table(bronze_fqn)

int_cols = [
    "WeekOfMonth", "WeekOfMonthClaimed", "Age",
    "FraudFound_P", "PolicyNumber", "RepNumber",
    "Deductible", "DriverRating", "Year",
]
for c in int_cols:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast(IntegerType()))

for c in [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]:
    df = df.withColumn(c, F.trim(F.col(c)))

df = df.where(F.col("PolicyNumber").isNotNull()).fillna({
    "Deductible":   0,
    "DriverRating": 0,
    "Age":          0,
    "FraudFound_P": 0,
})

order_col = "ingestion_timestamp" if "ingestion_timestamp" in df.columns else "PolicyNumber"
w = Window.partitionBy("PolicyNumber").orderBy(F.col(order_col).desc_nulls_last())
df = (df.withColumn("_rn", F.row_number().over(w))
        .where(F.col("_rn") == 1)
        .drop("_rn"))

df = (df.withColumn("silver_processed_at", F.current_timestamp())
        .withColumn("silver_run_date",     F.lit(run_date)))

# COMMAND ----------

if not spark.catalog.tableExists(silver_fqn):
    df.limit(0).write.format("delta").saveAsTable(silver_fqn)

df.createOrReplaceTempView("silver_source")

target_cols  = [c.name for c in spark.table(silver_fqn).schema.fields]
non_key_cols = [c for c in target_cols if c != "PolicyNumber" and c in df.columns]
update_set   = ", ".join([f"t.`{c}` = s.`{c}`" for c in non_key_cols])

spark.sql(f"""
MERGE INTO {silver_fqn} t
USING silver_source s
  ON t.PolicyNumber = s.PolicyNumber
WHEN MATCHED THEN UPDATE SET {update_set}
WHEN NOT MATCHED THEN INSERT *
""")

# COMMAND ----------

row_count    = spark.table(silver_fqn).count()
distinct_pks = spark.table(silver_fqn).select("PolicyNumber").distinct().count()
assert row_count == distinct_pks, f"dedup failed: {row_count} rows, {distinct_pks} distinct"

print(f"{silver_fqn} -> {row_count:,}")
display(spark.table(silver_fqn).limit(5))

# COMMAND ----------

dbutils.notebook.exit(f"silver_rows={row_count}")
