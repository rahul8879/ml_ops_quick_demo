# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Feature Engineering (Unity Catalog Feature Store)
# MAGIC
# MAGIC **Purpose:** Build modelling features from the silver claims table and register
# MAGIC them in Unity Catalog using `FeatureEngineeringClient` (the current, non-deprecated
# MAGIC API — `FeatureStoreClient` is deprecated).
# MAGIC
# MAGIC **Engineered features (minimum set required by spec):**
# MAGIC - `claim_to_premium_ratio` = ClaimAmount / AnnualPremium
# MAGIC - `vehicle_age`            = current_year − VehicleYear
# MAGIC - `high_deductible_flag`   = 1 if Deductible > 1000
# MAGIC - `days_since_policy_start` = datediff(ClaimDate, PolicyInceptionDate)
# MAGIC - `repeat_claimant_flag`   = 1 if same PolicyHolder has >1 prior claims
# MAGIC
# MAGIC **Inputs**
# MAGIC - `insurance_demo.silver.clean_claims`
# MAGIC
# MAGIC **Outputs**
# MAGIC - Feature table: `insurance_demo.gold.claim_features`  (PK = `PolicyNumber`)
# MAGIC
# MAGIC **Runtime:** Databricks Runtime 15.4 LTS ML

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",         "insurance_demo", "Unity Catalog")
dbutils.widgets.text("silver_schema",   "silver",         "Silver schema")
dbutils.widgets.text("silver_table",    "clean_claims",   "Silver table")
dbutils.widgets.text("gold_schema",     "gold",           "Gold schema")
dbutils.widgets.text("feature_table",   "claim_features", "Feature table")
dbutils.widgets.text("run_date",        datetime.utcnow().strftime("%Y-%m-%d"), "Run date (YYYY-MM-DD)")

catalog        = dbutils.widgets.get("catalog")
silver_schema  = dbutils.widgets.get("silver_schema")
silver_table   = dbutils.widgets.get("silver_table")
gold_schema    = dbutils.widgets.get("gold_schema")
feature_table  = dbutils.widgets.get("feature_table")
run_date       = dbutils.widgets.get("run_date")

silver_fqn  = f"{catalog}.{silver_schema}.{silver_table}"
feature_fqn = f"{catalog}.{gold_schema}.{feature_table}"

print(f"Silver source : {silver_fqn}")
print(f"Feature table : {feature_fqn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ensure gold schema exists

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {catalog}.{gold_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Engineer features

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

silver_df = spark.table(silver_fqn)

# --- Resolve the claim date column and the current year ------------------
# Some Kaggle dumps carry "DateOfIncident" instead of "ClaimDate"; we normalize.
claim_date_col = None
for candidate in ["ClaimDate", "DateOfIncident", "IncidentDate"]:
    if candidate in silver_df.columns:
        claim_date_col = candidate
        break
if claim_date_col is None:
    # Fall back to current_date so the pipeline still runs end-to-end in demos.
    silver_df = silver_df.withColumn("ClaimDate", F.current_date())
    claim_date_col = "ClaimDate"

current_year_lit = F.lit(int(run_date.split("-")[0]))

# --- Core ratios & flags -------------------------------------------------
features_df = (
    silver_df
        .withColumn(
            "claim_to_premium_ratio",
            F.when(F.col("AnnualPremium") > 0, F.col("ClaimAmount") / F.col("AnnualPremium"))
             .otherwise(F.lit(None).cast("double")),
        )
        .withColumn("vehicle_age",          (current_year_lit - F.col("VehicleYear")).cast("int"))
        .withColumn("high_deductible_flag", F.when(F.col("Deductible") > 1000, 1).otherwise(0).cast("int"))
        .withColumn(
            "days_since_policy_start",
            F.datediff(F.col(claim_date_col), F.col("PolicyInceptionDate")).cast("int"),
        )
)

# --- repeat_claimant_flag ------------------------------------------------
# Definition (per spec): 1 if the same PolicyHolder has > 1 prior claims in the
# dataset. We use PolicyHolderID if present, otherwise fall back to PolicyNumber.
holder_col = "PolicyHolderID" if "PolicyHolderID" in features_df.columns else "PolicyNumber"

prior_claims_w = (
    Window
      .partitionBy(holder_col)
      .orderBy(F.col(claim_date_col).asc_nulls_last())
      .rowsBetween(Window.unboundedPreceding, -1)
)

features_df = features_df.withColumn(
    "prior_claim_count",
    F.count("*").over(prior_claims_w).cast("int"),
)
features_df = features_df.withColumn(
    "repeat_claimant_flag",
    F.when(F.col("prior_claim_count") > 1, 1).otherwise(0).cast("int"),
)

# --- Helpful extras (kept minimal) ---------------------------------------
if "Age" in features_df.columns:
    features_df = features_df.withColumn(
        "driver_age_bucket",
        F.when(F.col("Age") < 25, "under_25")
         .when(F.col("Age") < 40, "25_to_39")
         .when(F.col("Age") < 60, "40_to_59")
         .otherwise("60_plus"),
    )

# --- Select feature-table columns ---------------------------------------
# Primary key must be present. Keep label alongside features for training,
# but Feature Store training set joins only pull the feature columns.
keep_cols = [
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
]
if "driver_age_bucket" in features_df.columns:
    keep_cols.append("driver_age_bucket")

# Only keep columns that actually exist to stay robust to schema drift.
keep_cols = [c for c in keep_cols if c in features_df.columns]
features_out = features_df.select(*keep_cols).dropna(subset=["PolicyNumber"]).dropDuplicates(["PolicyNumber"])

# Bookkeeping
features_out = features_out.withColumn("feature_computed_at", F.current_timestamp())

print(f"Feature row count: {features_out.count():,}")
features_out.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register / update the feature table with FeatureEngineeringClient
# MAGIC
# MAGIC `FeatureEngineeringClient` is the correct client for Unity-Catalog-backed
# MAGIC feature tables. The older `FeatureStoreClient` is deprecated on UC.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

table_exists = spark.catalog.tableExists(feature_fqn)

if not table_exists:
    fe.create_table(
        name=feature_fqn,
        primary_keys=["PolicyNumber"],
        df=features_out,
        description=(
            "Feature table for vehicle-insurance claim fraud detection. "
            "Row grain = one PolicyNumber."
        ),
        tags={
            "layer":   "gold",
            "domain":  "insurance_fraud",
            "owner":   "ds_team",
            "refresh": "daily",
        },
    )
    print(f"Created feature table {feature_fqn}")
else:
    # write_table with merge mode performs an upsert on primary key.
    fe.write_table(
        name=feature_fqn,
        df=features_out,
        mode="merge",
    )
    print(f"Merged feature updates into {feature_fqn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Apply Unity Catalog column comments & tags

# COMMAND ----------

column_docs = {
    "PolicyNumber":            ("Primary key — unique insurance policy identifier.",            {"pii": "false", "role": "primary_key"}),
    "claim_to_premium_ratio":  ("Ratio of ClaimAmount to AnnualPremium. Useful fraud signal.",  {"role": "feature", "fraud_signal": "high"}),
    "vehicle_age":             ("Age of the insured vehicle in years at claim year.",           {"role": "feature"}),
    "high_deductible_flag":    ("1 if Deductible > 1000; otherwise 0.",                         {"role": "feature", "type": "flag"}),
    "days_since_policy_start": ("Days between PolicyInceptionDate and ClaimDate.",              {"role": "feature"}),
    "repeat_claimant_flag":    ("1 if the policyholder has >1 prior claims in history.",        {"role": "feature", "fraud_signal": "medium"}),
    "prior_claim_count":       ("Count of prior claims by the same policyholder.",              {"role": "feature"}),
    "ClaimAmount":             ("Monetary claim amount (passthrough from silver).",             {"role": "feature"}),
    "AnnualPremium":           ("Annual insurance premium (passthrough).",                      {"role": "feature"}),
    "Deductible":              ("Policy deductible (passthrough).",                             {"role": "feature"}),
    "driver_age_bucket":       ("Bucketed driver age: under_25, 25_to_39, 40_to_59, 60_plus.",  {"role": "feature", "type": "categorical"}),
    "feature_computed_at":     ("Timestamp this feature row was last computed.",                {"role": "metadata"}),
}

current_cols = set(c.name for c in spark.table(feature_fqn).schema.fields)

for col, (comment, tags) in column_docs.items():
    if col not in current_cols:
        continue
    # Escape any single quotes in comments
    safe_comment = comment.replace("'", "''")
    spark.sql(
        f"ALTER TABLE {feature_fqn} ALTER COLUMN `{col}` COMMENT '{safe_comment}'"
    )
    tag_pairs = ", ".join([f"'{k}' = '{v}'" for k, v in tags.items()])
    spark.sql(
        f"ALTER TABLE {feature_fqn} ALTER COLUMN `{col}` SET TAGS ({tag_pairs})"
    )

# Table-level comment + tags
spark.sql(f"""
    COMMENT ON TABLE {feature_fqn} IS
    'Gold feature table for the insurance claim fraud detector. Primary key: PolicyNumber.'
""")
spark.sql(f"ALTER TABLE {feature_fqn} SET TAGS ('layer' = 'gold', 'domain' = 'insurance_fraud', 'pii' = 'false')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sanity checks

# COMMAND ----------

display(spark.table(feature_fqn).limit(10))

n_rows    = spark.table(feature_fqn).count()
n_unique  = spark.table(feature_fqn).select("PolicyNumber").distinct().count()
n_nulls   = spark.table(feature_fqn).where("claim_to_premium_ratio IS NULL").count()

assert n_rows == n_unique, f"Primary key not unique: rows={n_rows}, distinct={n_unique}"

print(f"feature_rows={n_rows}, distinct_keys={n_unique}, null_ratio_rows={n_nulls}")

# COMMAND ----------

dbutils.notebook.exit(f"feature_rows={n_rows}")
