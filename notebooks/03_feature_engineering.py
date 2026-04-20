# Databricks notebook source
# MAGIC %md # Feature engineering

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",       "wcqmlopsdemo",                        "catalog")
dbutils.widgets.text("silver_schema", "silver",                              "silver schema")
dbutils.widgets.text("silver_table",  "clean_claims",                        "silver table")
dbutils.widgets.text("gold_schema",   "gold",                                "gold schema")
dbutils.widgets.text("feature_table", "claim_features",                      "feature table")
dbutils.widgets.text("run_date",      datetime.utcnow().strftime("%Y-%m-%d"), "run date")

catalog       = dbutils.widgets.get("catalog")
silver_schema = dbutils.widgets.get("silver_schema")
silver_table  = dbutils.widgets.get("silver_table")
gold_schema   = dbutils.widgets.get("gold_schema")
feature_table = dbutils.widgets.get("feature_table")
run_date      = dbutils.widgets.get("run_date")

silver_fqn  = f"{catalog}.{silver_schema}.{silver_table}"
feature_fqn = f"{catalog}.{gold_schema}.{feature_table}"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{gold_schema}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

silver_df = spark.table(silver_fqn)

claim_date_col = None
for c in ["ClaimDate", "DateOfIncident", "IncidentDate"]:
    if c in silver_df.columns:
        claim_date_col = c
        break
if claim_date_col is None:
    silver_df = silver_df.withColumn("ClaimDate", F.current_date())
    claim_date_col = "ClaimDate"

current_year = F.lit(int(run_date.split("-")[0]))

features_df = (
    silver_df
      .withColumn(
          "claim_to_premium_ratio",
          F.when(F.col("AnnualPremium") > 0, F.col("ClaimAmount") / F.col("AnnualPremium"))
           .otherwise(F.lit(None).cast("double")),
      )
      .withColumn("vehicle_age",          (current_year - F.col("VehicleYear")).cast("int"))
      .withColumn("high_deductible_flag", F.when(F.col("Deductible") > 1000, 1).otherwise(0).cast("int"))
      .withColumn(
          "days_since_policy_start",
          F.datediff(F.col(claim_date_col), F.col("PolicyInceptionDate")).cast("int"),
      )
)

holder_col = "PolicyHolderID" if "PolicyHolderID" in features_df.columns else "PolicyNumber"

prior_w = (
    Window.partitionBy(holder_col)
          .orderBy(F.col(claim_date_col).asc_nulls_last())
          .rowsBetween(Window.unboundedPreceding, -1)
)

features_df = (
    features_df
      .withColumn("prior_claim_count",    F.count("*").over(prior_w).cast("int"))
      .withColumn("repeat_claimant_flag", F.when(F.col("prior_claim_count") > 1, 1).otherwise(0).cast("int"))
)

if "Age" in features_df.columns:
    features_df = features_df.withColumn(
        "driver_age_bucket",
        F.when(F.col("Age") < 25, "under_25")
         .when(F.col("Age") < 40, "25_to_39")
         .when(F.col("Age") < 60, "40_to_59")
         .otherwise("60_plus"),
    )

keep = [
    "PolicyNumber",
    "claim_to_premium_ratio",
    "vehicle_age",
    "high_deductible_flag",
    "days_since_policy_start",
    "repeat_claimant_flag",
    "prior_claim_count",
    "ClaimAmount",
    "AnnualPremium",
    "Deductible",
    "driver_age_bucket",
]
keep = [c for c in keep if c in features_df.columns]

features_out = (
    features_df.select(*keep)
               .dropna(subset=["PolicyNumber"])
               .dropDuplicates(["PolicyNumber"])
               .withColumn("feature_computed_at", F.current_timestamp())
)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

if not spark.catalog.tableExists(feature_fqn):
    fe.create_table(
        name=feature_fqn,
        primary_keys=["PolicyNumber"],
        df=features_out,
        description="Features for vehicle insurance claim fraud detection. Grain = PolicyNumber.",
        tags={"layer": "gold", "domain": "insurance_fraud", "owner": "ds_team", "refresh": "daily"},
    )
else:
    fe.write_table(name=feature_fqn, df=features_out, mode="merge")

# COMMAND ----------

column_docs = {
    "PolicyNumber":            ("Primary key.",                                    {"role": "primary_key"}),
    "claim_to_premium_ratio":  ("ClaimAmount / AnnualPremium.",                    {"role": "feature", "fraud_signal": "high"}),
    "vehicle_age":             ("Current year minus VehicleYear.",                 {"role": "feature"}),
    "high_deductible_flag":    ("1 if Deductible > 1000 else 0.",                  {"role": "feature", "type": "flag"}),
    "days_since_policy_start": ("Days between PolicyInceptionDate and ClaimDate.", {"role": "feature"}),
    "repeat_claimant_flag":    ("1 if policyholder has > 1 prior claims.",         {"role": "feature", "fraud_signal": "medium"}),
    "prior_claim_count":       ("Count of prior claims by policyholder.",          {"role": "feature"}),
    "ClaimAmount":             ("Raw claim amount.",                               {"role": "feature"}),
    "AnnualPremium":           ("Annual premium.",                                 {"role": "feature"}),
    "Deductible":              ("Policy deductible.",                              {"role": "feature"}),
    "driver_age_bucket":       ("Bucketed driver age.",                            {"role": "feature", "type": "categorical"}),
    "feature_computed_at":     ("Row computation timestamp.",                      {"role": "metadata"}),
}

cols = {c.name for c in spark.table(feature_fqn).schema.fields}
for col, (comment, tags) in column_docs.items():
    if col not in cols:
        continue
    safe = comment.replace("'", "''")
    spark.sql(f"ALTER TABLE {feature_fqn} ALTER COLUMN `{col}` COMMENT '{safe}'")
    pairs = ", ".join([f"'{k}' = '{v}'" for k, v in tags.items()])
    spark.sql(f"ALTER TABLE {feature_fqn} ALTER COLUMN `{col}` SET TAGS ({pairs})")

spark.sql(f"ALTER TABLE {feature_fqn} SET TAGS ('layer' = 'gold', 'domain' = 'insurance_fraud', 'pii' = 'false')")

# COMMAND ----------

n_rows   = spark.table(feature_fqn).count()
n_unique = spark.table(feature_fqn).select("PolicyNumber").distinct().count()
assert n_rows == n_unique, f"PK not unique: {n_rows} rows, {n_unique} distinct"

display(spark.table(feature_fqn).limit(10))
print(f"{feature_fqn} -> {n_rows:,}")

# COMMAND ----------

dbutils.notebook.exit(f"feature_rows={n_rows}")
