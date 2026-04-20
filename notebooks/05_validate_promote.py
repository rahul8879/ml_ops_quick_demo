# Databricks notebook source
# MAGIC %md # Validate + promote

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering xgboost
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",           "wcqmlopsdemo",                         "catalog")
dbutils.widgets.text("silver_schema",     "silver",                               "silver schema")
dbutils.widgets.text("silver_table",      "clean_claims",                         "silver table")
dbutils.widgets.text("gold_schema",       "gold",                                 "gold schema")
dbutils.widgets.text("feature_table",    "claim_features",                        "feature table")
dbutils.widgets.text("models_schema",    "models",                                "models schema")
dbutils.widgets.text("model_name",       "fraud_detector",                        "model name")
dbutils.widgets.text("monitoring_schema","monitoring",                            "monitoring schema")
dbutils.widgets.text("validation_table", "model_validation_log",                  "validation log table")
dbutils.widgets.text("run_date",         datetime.utcnow().strftime("%Y-%m-%d"),  "run date")
dbutils.widgets.text("min_f1",           "0.72",                                  "min F1")
dbutils.widgets.text("min_auc",          "0.80",                                  "min AUC")
dbutils.widgets.text("max_psi",          "0.20",                                  "max PSI")

catalog           = dbutils.widgets.get("catalog")
silver_schema     = dbutils.widgets.get("silver_schema")
silver_table      = dbutils.widgets.get("silver_table")
gold_schema       = dbutils.widgets.get("gold_schema")
feature_table     = dbutils.widgets.get("feature_table")
models_schema     = dbutils.widgets.get("models_schema")
model_name        = dbutils.widgets.get("model_name")
monitoring_schema = dbutils.widgets.get("monitoring_schema")
validation_table  = dbutils.widgets.get("validation_table")
run_date          = dbutils.widgets.get("run_date")
MIN_F1            = float(dbutils.widgets.get("min_f1"))
MIN_AUC           = float(dbutils.widgets.get("min_auc"))
MAX_PSI           = float(dbutils.widgets.get("max_psi"))

silver_fqn     = f"{catalog}.{silver_schema}.{silver_table}"
feature_fqn    = f"{catalog}.{gold_schema}.{feature_table}"
model_fqn      = f"{catalog}.{models_schema}.{model_name}"
validation_fqn = f"{catalog}.{monitoring_schema}.{validation_table}"

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

versions  = client.search_model_versions(f"name = '{model_fqn}'")
assert versions, f"no versions for {model_fqn}"
candidate = sorted(versions, key=lambda v: int(v.version))[-1]
candidate_version = int(candidate.version)

try:
    champion_mv = client.get_model_version_by_alias(model_fqn, "Champion")
    champion_version = int(champion_mv.version)
except Exception:
    champion_mv = None
    champion_version = None

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pandas as pd
from sklearn.model_selection import train_test_split

fe = FeatureEngineeringClient()

ground_truth_df = (
    spark.table(silver_fqn)
         .select("PolicyNumber", "FraudFound_P")
         .dropna(subset=["PolicyNumber", "FraudFound_P"])
)

feature_cols = [
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
feat_schema_cols = [c.name for c in spark.table(feature_fqn).schema.fields]
feature_cols = [c for c in feature_cols if c in feat_schema_cols]

training_set = fe.create_training_set(
    df=ground_truth_df,
    feature_lookups=[FeatureLookup(table_name=feature_fqn, lookup_key="PolicyNumber", feature_names=feature_cols)],
    label="FraudFound_P",
    exclude_columns=[],
)
full_pdf = training_set.load_df().toPandas()

y_all = full_pdf["FraudFound_P"].astype(int)
X_all = full_pdf.drop(columns=["FraudFound_P", "PolicyNumber"], errors="ignore")

cat_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]
if cat_cols:
    X_all = pd.get_dummies(X_all, columns=cat_cols, drop_first=False)
X_all = X_all.fillna(0.0).astype("float64")

_, X_holdout, _, y_holdout = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# COMMAND ----------

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def score_version(version: int):
    pyfunc = mlflow.pyfunc.load_model(f"models:/{model_fqn}/{version}")
    preds  = np.asarray(pyfunc.predict(X_holdout))
    if preds.ndim == 2:
        proba  = preds[:, 1]
        labels = (proba >= 0.5).astype(int)
    elif preds.dtype.kind == "f" and preds.min() >= 0.0 and preds.max() <= 1.0:
        proba  = preds
        labels = (proba >= 0.5).astype(int)
    else:
        labels = preds.astype(int)
        proba  = labels.astype(float)
    f1  = f1_score(y_holdout, labels)
    auc = roc_auc_score(y_holdout, proba) if len(set(y_holdout)) > 1 else float("nan")
    return float(f1), float(auc)

cand_f1, cand_auc = score_version(candidate_version)

# COMMAND ----------

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = pd.Series(expected).dropna().astype(float).values
    actual   = pd.Series(actual).dropna().astype(float).values
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    try:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
        if len(edges) < 3:
            edges = np.linspace(expected.min(), expected.max() + 1e-9, buckets + 1)
    except Exception:
        edges = np.linspace(expected.min(), expected.max() + 1e-9, buckets + 1)
    e, _ = np.histogram(expected, bins=edges)
    a, _ = np.histogram(actual,   bins=edges)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

baseline_pdf = spark.table(feature_fqn).toPandas()

psi_results = {}
for col in feature_cols:
    if col not in baseline_pdf.columns or col not in X_holdout.columns:
        continue
    if not np.issubdtype(baseline_pdf[col].dtype, np.number):
        continue
    psi_results[col] = psi(baseline_pdf[col].values, X_holdout[col].values)

max_psi_col = max(psi_results, key=psi_results.get) if psi_results else None
max_psi_val = psi_results[max_psi_col] if max_psi_col else 0.0

# COMMAND ----------

checks = {
    "f1_gate":  cand_f1       >= MIN_F1,
    "auc_gate": cand_auc      >= MIN_AUC,
    "psi_gate": max_psi_val   <= MAX_PSI,
}
passed_all = all(checks.values())
print(f"candidate v{candidate_version}  f1={cand_f1:.4f}  auc={cand_auc:.4f}  max_psi={max_psi_val:.4f}  checks={checks}")

# COMMAND ----------

promotion_action      = "none"
previous_champion_f1  = None

if passed_all:
    client.set_registered_model_alias(name=model_fqn, alias="Challenger", version=candidate_version)
    promotion_action = "set_challenger"

    if champion_version is None:
        client.set_registered_model_alias(name=model_fqn, alias="Champion", version=candidate_version)
        promotion_action = "seed_champion"
    else:
        champ_f1, _ = score_version(champion_version)
        previous_champion_f1 = champ_f1
        if cand_f1 > champ_f1:
            client.set_registered_model_alias(name=model_fqn, alias="Champion", version=candidate_version)
            promotion_action = "promote_to_champion"
        else:
            promotion_action = "kept_champion"

print(f"action={promotion_action}")

# COMMAND ----------

import json
from pyspark.sql import Row
from pyspark.sql import functions as F

row = Row(
    run_date              = run_date,
    evaluated_at          = datetime.utcnow(),
    model_fqn             = model_fqn,
    candidate_version     = candidate_version,
    candidate_run_id      = candidate.run_id,
    candidate_f1          = cand_f1,
    candidate_roc_auc     = cand_auc,
    max_psi               = max_psi_val,
    max_psi_feature       = max_psi_col or "n/a",
    psi_per_feature_json  = json.dumps(psi_results),
    f1_gate_pass          = bool(checks["f1_gate"]),
    auc_gate_pass         = bool(checks["auc_gate"]),
    psi_gate_pass         = bool(checks["psi_gate"]),
    all_gates_pass        = bool(passed_all),
    promotion_action      = promotion_action,
    previous_champion_version = int(champion_version) if champion_version is not None else None,
    previous_champion_f1  = float(previous_champion_f1) if previous_champion_f1 is not None else None,
)
audit_df = spark.createDataFrame([row])

audit_df.write.format("delta").mode("append").saveAsTable(validation_fqn)

display(spark.table(validation_fqn).orderBy(F.col("evaluated_at").desc()).limit(5))

# COMMAND ----------

dbutils.notebook.exit(
    f"candidate_version={candidate_version};f1={cand_f1:.4f};auc={cand_auc:.4f};action={promotion_action}"
)
