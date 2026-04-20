# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Transform (Clean + Dedup + CDF)
# MAGIC
# MAGIC **Purpose:** Read the Bronze claims table, apply type casting, deduplication on
# MAGIC `PolicyNumber`, and null-handling. Upsert into the Silver table using Delta
# MAGIC `MERGE`. Change Data Feed is enabled so downstream feature engineering can
# MAGIC consume incremental changes.
# MAGIC
# MAGIC **Inputs**
# MAGIC - Delta table: `insurance_demo.bronze.raw_claims`
# MAGIC
# MAGIC **Outputs**
# MAGIC - Delta table: `insurance_demo.silver.clean_claims` (CDF enabled, MERGE-ed on `PolicyNumber`)
# MAGIC
# MAGIC **Runtime:** Databricks Runtime 15.4 LTS ML

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets

# COMMAND ----------

dbutils.widgets.text("catalog",        "insurance_demo", "Unity Catalog")
dbutils.widgets.text("bronze_schema",  "bronze",         "Bronze schema")
dbutils.widgets.text("bronze_table",   "raw_claims",     "Bronze table")
dbutils.widgets.text("silver_schema",  "silver",         "Silver schema")
dbutils.widgets.text("silver_table",   "clean_claims",   "Silver table")
dbutils.widgets.text("run_date",       "",               "Run date (YYYY-MM-DD, optional)")

catalog        = dbutils.widgets.get("catalog")
bronze_schema  = dbutils.widgets.get("bronze_schema")
bronze_table   = dbutils.widgets.get("bronze_table")
silver_schema  = dbutils.widgets.get("silver_schema")
silver_table   = dbutils.widgets.get("silver_table")
run_date       = dbutils.widgets.get("run_date")

bronze_fqn = f"{catalog}.{bronze_schema}.{bronze_table}"
silver_fqn = f"{catalog}.{silver_schema}.{silver_table}"

print(f"Reading from : {bronze_fqn}")
print(f"Writing to   : {silver_fqn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ensure silver schema exists

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{silver_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Read Bronze, apply casting + cleaning

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, DateType, StringType
from pyspark.sql.window import Window

bronze_df = spark.table(bronze_fqn)

# ---- Type casting --------------------------------------------------------
# Kaggle "Vehicle Claim Fraud Detection" columns. Defensive casts: if a
# column is missing (e.g. schema drift), withColumn simply creates it as null
# then casts — errors are surfaced explicitly at the end.
numeric_int_cols = [
    "Age", "Deductible", "DriverRating", "Year", "VehicleYear",
    "RepNumber", "PolicyNumber", "WeekOfMonth", "WeekOfMonthClaimed",
    "FraudFound_P",
]
numeric_double_cols = ["ClaimAmount", "AnnualPremium"]

silver_df = bronze_df

for c in numeric_int_cols:
    if c in silver_df.columns:
        silver_df = silver_df.withColumn(c, F.col(c).cast(IntegerType()))

for c in numeric_double_cols:
    if c in silver_df.columns:
        silver_df = silver_df.withColumn(c, F.col(c).cast(DoubleType()))

# ClaimDate / PolicyInceptionDate may arrive as strings in several formats.
for c in ["ClaimDate", "PolicyInceptionDate", "DateOfIncident"]:
    if c in silver_df.columns:
        silver_df = silver_df.withColumn(
            c,
            F.coalesce(
                F.to_date(F.col(c), "yyyy-MM-dd"),
                F.to_date(F.col(c), "MM/dd/yyyy"),
                F.to_date(F.col(c), "dd-MM-yyyy"),
            ),
        )

# Trim whitespace on string columns
string_cols = [f.name for f in silver_df.schema.fields if isinstance(f.dataType, StringType)]
for c in string_cols:
    silver_df = silver_df.withColumn(c, F.trim(F.col(c)))

# ---- Null handling -------------------------------------------------------
# Drop rows missing the natural key.
silver_df = silver_df.where(F.col("PolicyNumber").isNotNull())

# Fill numeric nulls with sensible defaults (safer than dropping 6%-fraud rows).
silver_df = silver_df.fillna({
    "ClaimAmount":    0.0,
    "AnnualPremium":  0.0,
    "Deductible":     0,
    "DriverRating":   0,
    "FraudFound_P":   0,
})

# ---- Deduplication on PolicyNumber --------------------------------------
# Keep the most recent row per PolicyNumber (using ingestion_timestamp tie-breaker).
order_col = "ingestion_timestamp" if "ingestion_timestamp" in silver_df.columns else "PolicyNumber"
w = Window.partitionBy("PolicyNumber").orderBy(F.col(order_col).desc_nulls_last())
silver_df = (
    silver_df.withColumn("_rn", F.row_number().over(w))
             .where(F.col("_rn") == 1)
             .drop("_rn")
)

# Bookkeeping columns
silver_df = (
    silver_df
        .withColumn("silver_processed_at", F.current_timestamp())
        .withColumn("silver_run_date",     F.lit(run_date))
)

print(f"Silver dedup'd row count: {silver_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create silver table if not exists (CDF enabled)

# COMMAND ----------

# Use the in-memory dataframe's schema to create an empty target table with
# Change Data Feed turned on. We then MERGE into it — this gives us proper
# upsert semantics rather than overwrite.
if not spark.catalog.tableExists(silver_fqn):
    (
        silver_df.limit(0)
                 .write
                 .format("delta")
                 .option("delta.enableChangeDataFeed", "true")
                 .saveAsTable(silver_fqn)
    )
    print(f"Created silver table {silver_fqn} with CDF enabled.")
else:
    # Make sure CDF is enabled even if the table pre-existed.
    spark.sql(f"ALTER TABLE {silver_fqn} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MERGE (upsert) into silver on PolicyNumber

# COMMAND ----------

silver_df.createOrReplaceTempView("silver_source")

# Build the non-key column list dynamically so the MERGE survives schema evolution.
target_cols = [c.name for c in spark.table(silver_fqn).schema.fields]
non_key_cols = [c for c in target_cols if c != "PolicyNumber"]
update_set = ", ".join([f"t.`{c}` = s.`{c}`" for c in non_key_cols if c in silver_df.columns])

merge_sql = f"""
MERGE INTO {silver_fqn} t
USING silver_source s
  ON t.PolicyNumber = s.PolicyNumber
WHEN MATCHED THEN UPDATE SET {update_set}
WHEN NOT MATCHED THEN INSERT *
"""

spark.sql(merge_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Table properties, comment, tags

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {silver_fqn}
    SET TBLPROPERTIES (
        'delta.enableChangeDataFeed'       = 'true',
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact'   = 'true',
        'quality'                          = 'silver'
    )
""")

spark.sql(f"""
    COMMENT ON TABLE {silver_fqn} IS
    'Cleaned, deduplicated insurance claims. Upserted on PolicyNumber. CDF enabled for downstream feature engineering.'
""")

spark.sql(f"ALTER TABLE {silver_fqn} SET TAGS ('layer' = 'silver', 'domain' = 'insurance_fraud', 'cdf' = 'enabled')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validation

# COMMAND ----------

row_count = spark.table(silver_fqn).count()
distinct_policies = spark.table(silver_fqn).select("PolicyNumber").distinct().count()
assert row_count == distinct_policies, (
    f"Dedup invariant failed: rows={row_count}, distinct PolicyNumber={distinct_policies}"
)

print(f"[silver] {silver_fqn} row count = {row_count:,} (distinct PolicyNumber = {distinct_policies:,})")

display(spark.table(silver_fqn).limit(5))

# COMMAND ----------

dbutils.notebook.exit(f"silver_rows={row_count}")
