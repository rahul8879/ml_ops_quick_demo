# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Lakehouse Monitoring on the Inference Table
# MAGIC
# MAGIC **Purpose:** Attach a Lakehouse Monitor (InferenceLog profile) to the
# MAGIC inference-capture table produced by the serving endpoint, using the feature
# MAGIC table as the training distribution baseline. Prints the auto-generated
# MAGIC monitoring dashboard URL.
# MAGIC
# MAGIC **Inputs**
# MAGIC - Inference table : `insurance_demo.monitoring.fraud_inference_payload`
# MAGIC - Baseline table  : `insurance_demo.gold.claim_features`
# MAGIC
# MAGIC **Schedule:** hourly refresh

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk>=0.30.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets

# COMMAND ----------

dbutils.widgets.text("catalog",           "insurance_demo",          "Unity Catalog")
dbutils.widgets.text("monitoring_schema", "monitoring",              "Monitoring schema")
dbutils.widgets.text("inference_table",   "fraud_inference_payload", "Inference payload table")
dbutils.widgets.text("gold_schema",       "gold",                    "Gold schema")
dbutils.widgets.text("feature_table",     "claim_features",          "Feature/baseline table")
dbutils.widgets.text("model_fqn",         "insurance_demo.models.fraud_detector", "Served model FQN")
dbutils.widgets.text("granularity",       "1 hour",                  "Time window granularity")
dbutils.widgets.text("timestamp_col",     "timestamp_ms",            "Timestamp column in inference table")
dbutils.widgets.text("prediction_col",    "prediction",              "Prediction column name")
dbutils.widgets.text("label_col",         "",                        "Ground-truth label column (empty if none)")
dbutils.widgets.text("model_id_col",      "model_version",           "Model ID column name")

catalog           = dbutils.widgets.get("catalog")
monitoring_schema = dbutils.widgets.get("monitoring_schema")
inference_table   = dbutils.widgets.get("inference_table")
gold_schema       = dbutils.widgets.get("gold_schema")
feature_table     = dbutils.widgets.get("feature_table")
model_fqn         = dbutils.widgets.get("model_fqn")
granularity       = dbutils.widgets.get("granularity")
timestamp_col     = dbutils.widgets.get("timestamp_col")
prediction_col    = dbutils.widgets.get("prediction_col")
label_col         = dbutils.widgets.get("label_col") or None
model_id_col      = dbutils.widgets.get("model_id_col")

inference_fqn = f"{catalog}.{monitoring_schema}.{inference_table}"
baseline_fqn  = f"{catalog}.{gold_schema}.{feature_table}"
output_schema = f"{catalog}.{monitoring_schema}"

print(f"Inference table : {inference_fqn}")
print(f"Baseline table  : {baseline_fqn}")
print(f"Output schema   : {output_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the monitor via Databricks SDK quality_monitors API

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorCronSchedule,
)

w = WorkspaceClient()

# Hourly cron: minute 0 of every hour, UTC (safest default for serverless refresh)
cron_hourly = MonitorCronSchedule(
    quartz_cron_expression="0 0 * * * ?",
    timezone_id="UTC",
)

inference_log_cfg = MonitorInferenceLog(
    granularities=[granularity],
    timestamp_col=timestamp_col,
    model_id_col=model_id_col,
    prediction_col=prediction_col,
    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
    label_col=label_col,   # may be None
)

# Idempotent: create if absent, else update in place.
try:
    existing_monitor = w.quality_monitors.get(table_name=inference_fqn)
    print(f"Monitor already exists for {inference_fqn} — updating.")
    monitor = w.quality_monitors.update(
        table_name=inference_fqn,
        inference_log=inference_log_cfg,
        output_schema_name=output_schema,
        baseline_table_name=baseline_fqn,
        schedule=cron_hourly,
    )
except Exception as e:
    print(f"Creating new monitor ({e.__class__.__name__}).")
    monitor = w.quality_monitors.create(
        table_name=inference_fqn,
        inference_log=inference_log_cfg,
        output_schema_name=output_schema,
        baseline_table_name=baseline_fqn,
        schedule=cron_hourly,
    )

print("Monitor status:", monitor.status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Print the auto-generated dashboard URL

# COMMAND ----------

dashboard_id = getattr(monitor, "dashboard_id", None) or ""
workspace_url = spark.conf.get("spark.databricks.workspaceUrl", None)
if not workspace_url:
    workspace_url = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook().getContext().browserHostName().get()
    )

if dashboard_id:
    dashboard_url = f"https://{workspace_url}/sql/dashboardsv3/{dashboard_id}"
    print(f"Lakehouse Monitor dashboard: {dashboard_url}")
else:
    # Fallback: catalog-explorer view of the monitor for this table
    dashboard_url = (
        f"https://{workspace_url}/explore/data/{catalog}/"
        f"{monitoring_schema}/{inference_table}/quality"
    )
    print(f"Monitor page: {dashboard_url}")

print(f"Drift  metrics table : {monitor.drift_metrics_table_name}")
print(f"Profile metrics table: {monitor.profile_metrics_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Trigger an initial refresh so the dashboard has data on first load

# COMMAND ----------

try:
    refresh = w.quality_monitors.run_refresh(table_name=inference_fqn)
    print(f"Kicked off refresh id={refresh.refresh_id}, state={refresh.state}")
except Exception as e:
    # If the inference table has no rows yet, refresh will no-op. Don't fail.
    print(f"Refresh skipped: {e}")

# COMMAND ----------

dbutils.notebook.exit(f"monitor_table={inference_fqn};dashboard_url={dashboard_url}")
