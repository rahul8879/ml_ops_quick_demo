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

# Bucket → numeric midpoint mappings (Kaggle fraud_oracle.csv categorical columns)
VEHICLE_PRICE_MAP = {
    "less than 20000":  15000,
    "20000 to 29000":   25000,
    "30000 to 39000":   35000,
    "40000 to 59000":   50000,
    "60000 to 69000":   65000,
    "more than 69000":  80000,
}

AGE_OF_VEHICLE_MAP = {
    "new":          0,
    "2 years":      2,
    "3 years":      3,
    "4 years":      4,
    "5 years":      5,
    "6 years":      6,
    "7 years":      7,
    "more than 7":  8,
}

DAYS_POLICY_MAP = {
    "none":         0,
    "1 to 7":       4,
    "8 to 15":      12,
    "15 to 30":     22,
    "more than 30": 45,
}

PAST_CLAIMS_MAP = {
    "none":         0,
    "1":            1,
    "2 to 4":       3,
    "more than 4":  5,
}

df = silver_pdf.copy()

df["vehicle_price_numeric"]    = df["VehiclePrice"].map(VEHICLE_PRICE_MAP).fillna(0).astype(int)
df["vehicle_age"]              = df["AgeOfVehicle"].map(AGE_OF_VEHICLE_MAP).fillna(0).astype(int)
df["days_since_policy_start"]  = df["Days_Policy_Claim"].map(DAYS_POLICY_MAP).fillna(0).astype(int)
df["prior_claim_count"]        = df["PastNumberOfClaims"].map(PAST_CLAIMS_MAP).fillna(0).astype(int)

df["high_deductible_flag"]     = (df["Deductible"] > 500).astype(int)
df["repeat_claimant_flag"]     = df["PastNumberOfClaims"].isin(["2 to 4", "more than 4"]).astype(int)
df["no_police_report_flag"]    = (df["PoliceReportFiled"] == "No").astype(int)
df["no_witness_flag"]          = (df["WitnessPresent"] == "No").astype(int)
df["address_changed_flag"]     = (df["AddressChange_Claim"] != "no change").astype(int)
df["fault_policyholder_flag"]  = (df["Fault"] == "Policy Holder").astype(int)
df["internal_agent_flag"]      = (df["AgentType"] == "Internal").astype(int)

df["claim_to_premium_ratio"]   = df["vehicle_price_numeric"] / df["Deductible"].replace(0, pd.NA)
df["claim_to_premium_ratio"]   = df["claim_to_premium_ratio"].fillna(0).astype(float)

keep = [
    "PolicyNumber",
    "claim_to_premium_ratio",
    "vehicle_age",
    "high_deductible_flag",
    "days_since_policy_start",
    "repeat_claimant_flag",
    "prior_claim_count",
    "vehicle_price_numeric",
    "no_police_report_flag",
    "no_witness_flag",
    "address_changed_flag",
    "fault_policyholder_flag",
    "internal_agent_flag",
    "Deductible",
    "DriverRating",
    "Age",
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
