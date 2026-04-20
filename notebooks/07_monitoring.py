# Databricks notebook source
# MAGIC %md # Lakehouse monitor (processed inference table)

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk>=0.30.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog",           "wcqmlopsdemo",                     "catalog")
dbutils.widgets.text("monitoring_schema", "monitoring",                       "monitoring schema")
dbutils.widgets.text("payload_table",     "fraud_inference_payload_payload",  "AI gateway payload table")
dbutils.widgets.text("processed_table",   "fraud_inference_processed",        "processed inference table")
dbutils.widgets.text("gold_schema",       "gold",                             "gold schema")
dbutils.widgets.text("feature_table",     "claim_features",                   "baseline table")
dbutils.widgets.text("granularity",       "1 hour",                           "granularity")
dbutils.widgets.text("assets_dir",        "/Workspace/Shared/lakehouse_monitoring/fraud_inference", "assets dir")

catalog           = dbutils.widgets.get("catalog")
monitoring_schema = dbutils.widgets.get("monitoring_schema")
payload_table     = dbutils.widgets.get("payload_table")
processed_table   = dbutils.widgets.get("processed_table")
gold_schema       = dbutils.widgets.get("gold_schema")
feature_table     = dbutils.widgets.get("feature_table")
granularity       = dbutils.widgets.get("granularity")
assets_dir        = dbutils.widgets.get("assets_dir")

payload_fqn   = f"{catalog}.{monitoring_schema}.{payload_table}"
processed_fqn = f"{catalog}.{monitoring_schema}.{processed_table}"
baseline_fqn  = f"{catalog}.{gold_schema}.{feature_table}"
output_schema = f"{catalog}.{monitoring_schema}"

# COMMAND ----------

# MAGIC %md ### Build processed inference table from the AI-Gateway payload

# COMMAND ----------

payload_exists = spark.catalog.tableExists(payload_fqn)
print(f"payload exists: {payload_exists}")

if payload_exists:
    payload_cols = {c.name for c in spark.table(payload_fqn).schema.fields}
    print(f"payload cols: {sorted(payload_cols)}")

    if "timestamp_ms" in payload_cols:
        ts_expr = "CAST(timestamp_ms AS BIGINT)"
    elif "request_time" in payload_cols:
        ts_expr = "CAST(unix_millis(request_time) AS BIGINT)"
    elif "request_date" in payload_cols:
        ts_expr = "CAST(unix_millis(CAST(request_date AS TIMESTAMP)) AS BIGINT)"
    else:
        ts_expr = "CAST(unix_millis(current_timestamp()) AS BIGINT)"

    if "served_entity_name" in payload_cols:
        model_id_expr = "served_entity_name"
    elif "served_entity_id" in payload_cols:
        model_id_expr = "served_entity_id"
    else:
        model_id_expr = "'unknown'"

    prediction_expr = (
        "CAST(get_json_object(response, '$.predictions[0]') AS DOUBLE)"
        if "response" in payload_cols
        else "CAST(NULL AS DOUBLE)"
    )

    filter_parts = []
    if "status_code" in payload_cols:
        filter_parts.append("status_code = 200")
    if "response" in payload_cols:
        filter_parts.append("response IS NOT NULL")
    where_clause = "WHERE " + " AND ".join(filter_parts) if filter_parts else ""

    print(f"ts_expr        : {ts_expr}")
    print(f"model_id_expr  : {model_id_expr}")
    print(f"prediction_expr: {prediction_expr}")
    print(f"where_clause   : {where_clause}")

    spark.sql(f"""
        CREATE OR REPLACE TABLE {processed_fqn}
        USING DELTA
        AS
        SELECT
            {ts_expr}                        AS timestamp_ms,
            CAST({model_id_expr} AS STRING)  AS model_version,
            {prediction_expr}                AS prediction,
            CAST(NULL AS INT)                AS label
        FROM {payload_fqn}
        {where_clause}
    """)
else:
    print(f"{payload_fqn} not found — creating empty processed table with the expected schema.")
    from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, IntegerType
    schema = StructType([
        StructField("timestamp_ms",  LongType(),    True),
        StructField("model_version", StringType(),  True),
        StructField("prediction",    DoubleType(),  True),
        StructField("label",         IntegerType(), True),
    ])
    spark.createDataFrame([], schema) \
         .write.format("delta").mode("overwrite").saveAsTable(processed_fqn)

row_count = spark.table(processed_fqn).count()
print(f"{processed_fqn} -> {row_count:,} rows")

# seed one synthetic row if empty so the monitor has a schema to work with
if row_count == 0:
    import time
    spark.sql(f"""
        INSERT INTO {processed_fqn} VALUES
        ({int(time.time()*1000)}, 'bootstrap', 0.0, 0)
    """)
    print("inserted one bootstrap row")

display(spark.table(processed_fqn).limit(5))

# COMMAND ----------

# MAGIC %md ### Create (or update) the Lakehouse Monitor

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorCronSchedule,
)

w = WorkspaceClient()

cron_hourly = MonitorCronSchedule(quartz_cron_expression="0 0 * * * ?", timezone_id="UTC")

inference_log_cfg = MonitorInferenceLog(
    granularities=[granularity],
    timestamp_col="timestamp_ms",
    model_id_col="model_version",
    prediction_col="prediction",
    label_col="label",
    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
)

try:
    w.quality_monitors.get(table_name=processed_fqn)
    monitor = w.quality_monitors.update(
        table_name=processed_fqn,
        output_schema_name=output_schema,
        inference_log=inference_log_cfg,
        baseline_table_name=baseline_fqn,
        schedule=cron_hourly,
    )
except Exception:
    monitor = w.quality_monitors.create(
        table_name=processed_fqn,
        output_schema_name=output_schema,
        assets_dir=assets_dir,
        inference_log=inference_log_cfg,
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
    dashboard_url = f"https://{workspace_url}/explore/data/{catalog}/{monitoring_schema}/{processed_table}/quality"

print(f"dashboard:      {dashboard_url}")
print(f"drift table:    {monitor.drift_metrics_table_name}")
print(f"profile table:  {monitor.profile_metrics_table_name}")

# COMMAND ----------

try:
    refresh = w.quality_monitors.run_refresh(table_name=processed_fqn)
    print(f"refresh id={refresh.refresh_id} state={refresh.state}")
except Exception as e:
    print(f"refresh skipped: {e}")

# COMMAND ----------

dbutils.notebook.exit(f"monitor_table={processed_fqn};dashboard_url={dashboard_url}")
