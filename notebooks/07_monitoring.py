# Databricks notebook source
# MAGIC %md # Lakehouse monitor

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk>=0.30.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog",           "wcqmlopsdemo",              "catalog")
dbutils.widgets.text("monitoring_schema", "monitoring",                "monitoring schema")
dbutils.widgets.text("inference_table",   "fraud_inference_payload",   "inference table")
dbutils.widgets.text("gold_schema",       "gold",                      "gold schema")
dbutils.widgets.text("feature_table",     "claim_features",            "feature/baseline table")
dbutils.widgets.text("granularity",       "1 hour",                    "granularity")
dbutils.widgets.text("timestamp_col",     "timestamp_ms",              "timestamp column")
dbutils.widgets.text("prediction_col",    "prediction",                "prediction column")
dbutils.widgets.text("label_col",         "",                          "label column")
dbutils.widgets.text("model_id_col",      "model_version",             "model id column")

catalog           = dbutils.widgets.get("catalog")
monitoring_schema = dbutils.widgets.get("monitoring_schema")
inference_table   = dbutils.widgets.get("inference_table")
gold_schema       = dbutils.widgets.get("gold_schema")
feature_table     = dbutils.widgets.get("feature_table")
granularity       = dbutils.widgets.get("granularity")
timestamp_col     = dbutils.widgets.get("timestamp_col")
prediction_col    = dbutils.widgets.get("prediction_col")
label_col         = dbutils.widgets.get("label_col") or None
model_id_col      = dbutils.widgets.get("model_id_col")

inference_fqn = f"{catalog}.{monitoring_schema}.{inference_table}"
baseline_fqn  = f"{catalog}.{gold_schema}.{feature_table}"
output_schema = f"{catalog}.{monitoring_schema}"

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorCronSchedule,
)

w = WorkspaceClient()

cron_hourly = MonitorCronSchedule(quartz_cron_expression="0 0 * * * ?", timezone_id="UTC")

inference_log_cfg = MonitorInferenceLog(
    granularities=[granularity],
    timestamp_col=timestamp_col,
    model_id_col=model_id_col,
    prediction_col=prediction_col,
    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
    label_col=label_col,
)

try:
    w.quality_monitors.get(table_name=inference_fqn)
    monitor = w.quality_monitors.update(
        table_name=inference_fqn,
        inference_log=inference_log_cfg,
        output_schema_name=output_schema,
        baseline_table_name=baseline_fqn,
        schedule=cron_hourly,
    )
except Exception:
    monitor = w.quality_monitors.create(
        table_name=inference_fqn,
        inference_log=inference_log_cfg,
        output_schema_name=output_schema,
        baseline_table_name=baseline_fqn,
        schedule=cron_hourly,
    )

# COMMAND ----------

dashboard_id  = getattr(monitor, "dashboard_id", None) or ""
workspace_url = spark.conf.get("spark.databricks.workspaceUrl", None) \
    or dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

if dashboard_id:
    dashboard_url = f"https://{workspace_url}/sql/dashboardsv3/{dashboard_id}"
else:
    dashboard_url = f"https://{workspace_url}/explore/data/{catalog}/{monitoring_schema}/{inference_table}/quality"

print(f"dashboard:      {dashboard_url}")
print(f"drift table:    {monitor.drift_metrics_table_name}")
print(f"profile table:  {monitor.profile_metrics_table_name}")

# COMMAND ----------

try:
    refresh = w.quality_monitors.run_refresh(table_name=inference_fqn)
    print(f"refresh id={refresh.refresh_id} state={refresh.state}")
except Exception as e:
    print(f"refresh skipped: {e}")

# COMMAND ----------

dbutils.notebook.exit(f"monitor_table={inference_fqn};dashboard_url={dashboard_url}")
