# Databricks notebook source
# MAGIC %md # Validate + promote

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",          "wcqmlopsdemo",          "catalog")
dbutils.widgets.text("models_schema",    "models",                "models schema")
dbutils.widgets.text("model_name",       "fraud_detector",        "model name")
dbutils.widgets.text("monitoring_schema","monitoring",            "monitoring schema")
dbutils.widgets.text("validation_table", "model_validation_log",  "validation table")
dbutils.widgets.text("run_date",         datetime.utcnow().strftime("%Y-%m-%d"), "run date")
dbutils.widgets.text("min_f1",           "0.72",                  "min F1")
dbutils.widgets.text("min_auc",          "0.80",                  "min AUC")

catalog           = dbutils.widgets.get("catalog")
models_schema     = dbutils.widgets.get("models_schema")
model_name        = dbutils.widgets.get("model_name")
monitoring_schema = dbutils.widgets.get("monitoring_schema")
validation_table  = dbutils.widgets.get("validation_table")
run_date          = dbutils.widgets.get("run_date")
MIN_F1            = float(dbutils.widgets.get("min_f1"))
MIN_AUC           = float(dbutils.widgets.get("min_auc"))

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

run_metrics    = client.get_run(candidate.run_id).data.metrics
candidate_f1   = float(run_metrics.get("holdout_f1",      0.0))
candidate_auc  = float(run_metrics.get("holdout_roc_auc", 0.0))

print(f"candidate v{candidate_version}  f1={candidate_f1:.4f}  auc={candidate_auc:.4f}")

# COMMAND ----------

try:
    champion_mv = client.get_model_version_by_alias(model_fqn, "Champion")
    champion_version = int(champion_mv.version)
    champion_metrics = client.get_run(champion_mv.run_id).data.metrics
    champion_f1 = float(champion_metrics.get("holdout_f1", 0.0))
except Exception:
    champion_version = None
    champion_f1 = None

print(f"champion v{champion_version}  f1={champion_f1}")

# COMMAND ----------

f1_gate_pass  = candidate_f1  >= MIN_F1
auc_gate_pass = candidate_auc >= MIN_AUC
all_gates_pass = f1_gate_pass and auc_gate_pass

promotion_action = "none"

if all_gates_pass:
    client.set_registered_model_alias(name=model_fqn, alias="Challenger", version=candidate_version)
    promotion_action = "set_challenger"

    if champion_version is None:
        client.set_registered_model_alias(name=model_fqn, alias="Champion", version=candidate_version)
        promotion_action = "seed_champion"
    elif candidate_f1 > champion_f1:
        client.set_registered_model_alias(name=model_fqn, alias="Champion", version=candidate_version)
        promotion_action = "promote_to_champion"
    else:
        promotion_action = "kept_champion"

print(f"action={promotion_action}  gates={{f1:{f1_gate_pass}, auc:{auc_gate_pass}}}")

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

audit = {
    "run_date":                  run_date,
    "evaluated_at":              datetime.utcnow(),
    "model_fqn":                 model_fqn,
    "candidate_version":         int(candidate_version),
    "candidate_run_id":          candidate.run_id,
    "candidate_f1":              float(candidate_f1),
    "candidate_roc_auc":         float(candidate_auc),
    "f1_gate_pass":              bool(f1_gate_pass),
    "auc_gate_pass":             bool(auc_gate_pass),
    "all_gates_pass":            bool(all_gates_pass),
    "promotion_action":          promotion_action,
    "previous_champion_version": int(champion_version) if champion_version is not None else -1,
    "previous_champion_f1":      float(champion_f1)    if champion_f1      is not None else 0.0,
}

audit_pdf = pd.DataFrame([audit])
audit_df  = spark.createDataFrame(audit_pdf)

audit_df.write.format("delta").mode("append").saveAsTable(validation_fqn)

display(spark.table(validation_fqn).orderBy(F.col("evaluated_at").desc()).limit(5))

# COMMAND ----------

dbutils.notebook.exit(
    f"candidate_version={candidate_version};f1={candidate_f1:.4f};auc={candidate_auc:.4f};action={promotion_action}"
)
