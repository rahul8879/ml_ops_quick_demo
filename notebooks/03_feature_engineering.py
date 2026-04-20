# Databricks notebook source
# MAGIC %md # Feature engineering

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",       "wcqmlopsdemo",                         "catalog")
dbutils.widgets.text("silver_schema", "silver",                               "silver schema")
dbutils.widgets.text("silver_table",  "clean_claims",                         "silver table")
dbutils.widgets.text("gold_schema",   "gold",                                 "gold schema")
dbutils.widgets.text("feature_table", "claim_features",                       "feature table")
dbutils.widgets.text("run_date",      datetime.utcnow().strftime("%Y-%m-%d"), "run date")

catalog       = dbutils.widgets.get("catalog")
silver_schema = dbutils.widgets.get("silver_schema")
silver_table  = dbutils.widgets.get("silver_table")
gold_schema   = dbutils.widgets.get("gold_schema")
feature_table = dbutils.widgets.get("feature_table")
run_date      = dbutils.widgets.get("run_date")

silver_fqn  = f"{catalog}.{silver_schema}.{silver_table}"
feature_fqn = f"{catalog}.{gold_schema}.{feature_table}"

# COMMAND ----------

import pandas as pd

silver_pdf = spark.table(silver_fqn).toPandas()

claim_date_col = None
for c in ["ClaimDate", "DateOfIncident", "IncidentDate"]:
    if c in silver_pdf.columns:
        claim_date_col = c
        break
if claim_date_col is None:
    silver_pdf["ClaimDate"] = pd.Timestamp.utcnow().normalize()
    claim_date_col = "ClaimDate"

silver_pdf[claim_date_col]       = pd.to_datetime(silver_pdf[claim_date_col],       errors="coerce")
silver_pdf["PolicyInceptionDate"] = pd.to_datetime(silver_pdf.get("PolicyInceptionDate"), errors="coerce")

current_year = int(run_date.split("-")[0])

df = silver_pdf.copy()
df["claim_to_premium_ratio"] = df["ClaimAmount"] / df["AnnualPremium"].replace(0, pd.NA)
df["vehicle_age"]            = (current_year - df["VehicleYear"]).astype("Int64")
df["high_deductible_flag"]   = (df["Deductible"] > 1000).astype(int)
df["days_since_policy_start"] = (df[claim_date_col] - df["PolicyInceptionDate"]).dt.days.astype("Int64")

holder_col = "PolicyHolderID" if "PolicyHolderID" in df.columns else "PolicyNumber"
df = df.sort_values([holder_col, claim_date_col])
df["prior_claim_count"]    = df.groupby(holder_col).cumcount()
df["repeat_claimant_flag"] = (df["prior_claim_count"] > 1).astype(int)

if "Age" in df.columns:
    df["driver_age_bucket"] = pd.cut(
        df["Age"],
        bins=[-1, 24, 39, 59, 200],
        labels=["under_25", "25_to_39", "40_to_59", "60_plus"],
    ).astype(str)

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
keep = [c for c in keep if c in df.columns]

features_pdf = (
    df[keep]
      .dropna(subset=["PolicyNumber"])
      .drop_duplicates(subset=["PolicyNumber"])
      .reset_index(drop=True)
)
features_pdf["feature_computed_at"] = pd.Timestamp.utcnow()

features_df = spark.createDataFrame(features_pdf)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

if not spark.catalog.tableExists(feature_fqn):
    fe.create_table(
        name=feature_fqn,
        primary_keys=["PolicyNumber"],
        df=features_df,
        description="Features for vehicle insurance claim fraud detection.",
    )
else:
    fe.write_table(name=feature_fqn, df=features_df, mode="merge")

# COMMAND ----------

n_rows   = spark.table(feature_fqn).count()
n_unique = spark.table(feature_fqn).select("PolicyNumber").distinct().count()
assert n_rows == n_unique, f"PK not unique: {n_rows} rows, {n_unique} distinct"

display(spark.table(feature_fqn).limit(10))
print(f"{feature_fqn} -> {n_rows:,}")

# COMMAND ----------

dbutils.notebook.exit(f"feature_rows={n_rows}")
