# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Validate + Promote (Unity Catalog aliases)
# MAGIC
# MAGIC **Purpose:** Pull the newest registered version of
# MAGIC `insurance_demo.models.fraud_detector`, re-score it on a holdout slice,
# MAGIC evaluate against gating thresholds, check feature PSI drift vs. training
# MAGIC baseline, then manage UC aliases `Challenger` / `Champion`.
# MAGIC
# MAGIC - Validation gates
# MAGIC   - `F1 >= 0.72`
# MAGIC   - `ROC-AUC >= 0.80`
# MAGIC   - No feature's PSI vs. training baseline exceeds `0.2`
# MAGIC - If all gates pass → assign alias `Challenger`.
# MAGIC - If a `Champion` exists → compare F1; if the new candidate beats it → promote to `Champion`.
# MAGIC
# MAGIC **Outputs**
# MAGIC - Validation audit log: Delta table `insurance_demo.monitoring.model_validation_log`
# MAGIC - Alias changes on the UC model

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering xgboost
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",          "insurance_demo",  "Unity Catalog")
dbutils.widgets.text("silver_schema",    "silver",          "Silver schema")
dbutils.widgets.text("silver_table",     "clean_claims",    "Silver table (labels)")
dbutils.widgets.text("gold_schema",      "gold",            "Gold schema")
dbutils.widgets.text("feature_table",    "claim_features",  "Feature table")
dbutils.widgets.text("models_schema",    "models",          "Models schema")
dbutils.widgets.text("model_name",       "fraud_detector",  "Model name")
dbutils.widgets.text("monitoring_schema","monitoring",      "Monitoring schema")
dbutils.widgets.text("validation_table", "model_validation_log", "Validation log table")
dbutils.widgets.text("run_date",         datetime.utcnow().strftime("%Y-%m-%d"), "Run date")
dbutils.widgets.text("min_f1",           "0.72",            "Min F1 gate")
dbutils.widgets.text("min_auc",          "0.80",            "Min ROC-AUC gate")
dbutils.widgets.text("max_psi",          "0.20",            "Max per-feature PSI")

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

silver_fqn      = f"{catalog}.{silver_schema}.{silver_table}"
feature_fqn     = f"{catalog}.{gold_schema}.{feature_table}"
model_fqn       = f"{catalog}.{models_schema}.{model_name}"
validation_fqn  = f"{catalog}.{monitoring_schema}.{validation_table}"

print(f"Model    : {model_fqn}")
print(f"Gates    : F1 >= {MIN_F1}, AUC >= {MIN_AUC}, PSI <= {MAX_PSI}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure MLflow for UC & locate the latest version

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Ensure monitoring schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{monitoring_schema}")

# Newest version = max(version) for this registered model name
all_versions = client.search_model_versions(f"name = '{model_fqn}'")
assert len(all_versions) > 0, f"No versions found for {model_fqn}"
candidate = sorted(all_versions, key=lambda v: int(v.version))[-1]
candidate_version = int(candidate.version)

print(f"Candidate version: {candidate_version}  (run_id={candidate.run_id})")

# Is there an existing Champion?
try:
    champion_mv = client.get_model_version_by_alias(model_fqn, "Champion")
    champion_version = int(champion_mv.version)
    print(f"Current Champion : v{champion_version}")
except Exception:
    champion_mv = None
    champion_version = None
    print("No current Champion — this run will seed it if gates pass.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build a holdout set from Feature Store + labels

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pandas as pd

fe = FeatureEngineeringClient()

ground_truth_df = (
    spark.table(silver_fqn)
         .select("PolicyNumber", "FraudFound_P")
         .dropna(subset=["PolicyNumber", "FraudFound_P"])
)

feat_schema_cols = [c.name for c in spark.table(feature_fqn).schema.fields]
feature_cols = [
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
feature_cols = [c for c in feature_cols if c in feat_schema_cols]

lookups = [FeatureLookup(table_name=feature_fqn, lookup_key="PolicyNumber",
                         feature_names=feature_cols)]

training_set = fe.create_training_set(
    df=ground_truth_df,
    feature_lookups=lookups,
    label="FraudFound_P",
    exclude_columns=[],
)
full_pdf = training_set.load_df().toPandas()

# Deterministic holdout (same seed as training notebook)
from sklearn.model_selection import train_test_split
y_all = full_pdf["FraudFound_P"].astype(int)
X_all = full_pdf.drop(columns=["FraudFound_P", "PolicyNumber"], errors="ignore")

_, X_holdout, _, y_holdout = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

print(f"Holdout shape: {X_holdout.shape}, positives={int(y_holdout.sum())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score candidate model on holdout

# COMMAND ----------

from sklearn.metrics import f1_score, roc_auc_score

def score_version(version: int):
    """Return (f1, auc) for a given UC model version on the current holdout."""
    # Use fe.score_batch so it rehydrates features via the training set metadata.
    # For a simple in-memory score we just use the sklearn/xgboost flavor.
    model_uri = f"models:/{model_fqn}/{version}"
    pyfunc    = mlflow.pyfunc.load_model(model_uri)
    preds     = pyfunc.predict(X_holdout)
    # Some XGB wrappers return probs, some return class labels. Handle both.
    import numpy as np
    preds = np.asarray(preds)
    if preds.ndim == 2:
        proba = preds[:, 1]
        labels = (proba >= 0.5).astype(int)
    else:
        # If values are in [0,1] treat as probability, otherwise as labels.
        if preds.dtype.kind == "f" and preds.min() >= 0.0 and preds.max() <= 1.0:
            proba  = preds
            labels = (proba >= 0.5).astype(int)
        else:
            labels = preds.astype(int)
            proba  = labels.astype(float)
    f1  = f1_score(y_holdout, labels)
    auc = roc_auc_score(y_holdout, proba) if len(set(y_holdout)) > 1 else float("nan")
    return float(f1), float(auc)

cand_f1, cand_auc = score_version(candidate_version)
print(f"Candidate v{candidate_version} -> F1={cand_f1:.4f}, ROC-AUC={cand_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. PSI drift check vs. training baseline

# COMMAND ----------

import numpy as np

def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute PSI between two 1-D numeric arrays. Uses quantile-based bins on
    `expected`. PSI > 0.2 is typically treated as meaningful drift."""
    expected = pd.Series(expected).dropna().astype(float).values
    actual   = pd.Series(actual).dropna().astype(float).values
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    # Unique-quantile bin edges; fall back to equi-width if expected is constant.
    try:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
        if len(edges) < 3:
            edges = np.linspace(expected.min(), expected.max() + 1e-9, buckets + 1)
    except Exception:
        edges = np.linspace(expected.min(), expected.max() + 1e-9, buckets + 1)

    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual,   bins=edges)

    # Avoid div-by-zero
    exp_prop = np.clip(exp_hist / exp_hist.sum(), 1e-6, None)
    act_prop = np.clip(act_hist / act_hist.sum(), 1e-6, None)
    return float(np.sum((act_prop - exp_prop) * np.log(act_prop / exp_prop)))

# The baseline is the feature-table training distribution.
baseline_pdf = spark.table(feature_fqn).toPandas()

psi_results = {}
for col in feature_cols:
    if col not in baseline_pdf.columns or col not in X_holdout.columns:
        continue
    if not np.issubdtype(baseline_pdf[col].dtype, np.number):
        continue
    psi_results[col] = population_stability_index(
        baseline_pdf[col].values, X_holdout[col].values
    )

max_psi_col = max(psi_results, key=psi_results.get) if psi_results else None
max_psi_val = psi_results[max_psi_col] if max_psi_col else 0.0

print("PSI per feature:")
for c, v in sorted(psi_results.items(), key=lambda kv: -kv[1]):
    print(f"  {c:30s} PSI={v:.4f}")
print(f"Max PSI: {max_psi_val:.4f} ({max_psi_col})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Gate decision

# COMMAND ----------

checks = {
    "f1_gate":  cand_f1  >= MIN_F1,
    "auc_gate": cand_auc >= MIN_AUC,
    "psi_gate": max_psi_val <= MAX_PSI,
}
passed_all = all(checks.values())
print(f"Checks: {checks} → passed_all={passed_all}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Alias management (Challenger / Champion)

# COMMAND ----------

promotion_action = "none"
previous_champion_f1 = None

if passed_all:
    client.set_registered_model_alias(
        name=model_fqn, alias="Challenger", version=candidate_version
    )
    promotion_action = "set_challenger"
    print(f"Assigned Challenger alias to v{candidate_version}.")

    if champion_version is None:
        # No Champion yet — promote straight away.
        client.set_registered_model_alias(
            name=model_fqn, alias="Champion", version=candidate_version
        )
        promotion_action = "seed_champion"
        print(f"Promoted v{candidate_version} to Champion (first Champion).")
    else:
        # Compare on F1 using the SAME holdout.
        champ_f1, champ_auc = score_version(champion_version)
        previous_champion_f1 = champ_f1
        print(f"Champion v{champion_version} F1={champ_f1:.4f}, AUC={champ_auc:.4f}")
        if cand_f1 > champ_f1:
            client.set_registered_model_alias(
                name=model_fqn, alias="Champion", version=candidate_version
            )
            promotion_action = "promote_to_champion"
            print(f"Promoted v{candidate_version} to Champion (F1 beat {champ_f1:.4f}).")
        else:
            promotion_action = "kept_champion"
            print(f"Kept existing Champion v{champion_version}.")
else:
    print("Candidate failed gates — no alias changes applied.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Append audit row to monitoring Delta table

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
    max_psi_feature       = max_psi_col if max_psi_col else "n/a",
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

if not spark.catalog.tableExists(validation_fqn):
    (audit_df.write.format("delta").saveAsTable(validation_fqn))
    spark.sql(
        f"COMMENT ON TABLE {validation_fqn} IS "
        f"'Validation audit log for {model_fqn}. One row per validate-and-promote run.'"
    )
    spark.sql(
        f"ALTER TABLE {validation_fqn} SET TAGS "
        f"('layer' = 'monitoring', 'domain' = 'insurance_fraud')"
    )
else:
    audit_df.write.format("delta").mode("append").saveAsTable(validation_fqn)

display(spark.table(validation_fqn).orderBy(F.col("evaluated_at").desc()).limit(5))

# COMMAND ----------

dbutils.notebook.exit(
    f"candidate_version={candidate_version};f1={cand_f1:.4f};auc={cand_auc:.4f};action={promotion_action}"
)
