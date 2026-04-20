# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Deploy Model to Mosaic AI Model Serving
# MAGIC
# MAGIC **Purpose:** Stand up (or update in-place) a Mosaic AI Model Serving endpoint
# MAGIC named `fraud-detection-endpoint` that serves
# MAGIC `insurance_demo.models.fraud_detector@Champion`. Uses the **Databricks SDK**
# MAGIC (`WorkspaceClient`) — no raw REST calls — and wires an inference table for
# MAGIC downstream Lakehouse Monitoring.
# MAGIC
# MAGIC **Config**
# MAGIC - `workload_size`           : `Small`
# MAGIC - `scale_to_zero_enabled`   : `True` (cost control for demo)
# MAGIC - `auto_capture_config`     : inference table
# MAGIC   `insurance_demo.monitoring.fraud_inference_payload`
# MAGIC - Traffic: 100% → Champion

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk>=0.30.0 mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets

# COMMAND ----------

dbutils.widgets.text("catalog",          "insurance_demo",           "Unity Catalog")
dbutils.widgets.text("models_schema",    "models",                   "Models schema")
dbutils.widgets.text("model_name",       "fraud_detector",           "Model name")
dbutils.widgets.text("alias",            "Champion",                 "Alias to serve")
dbutils.widgets.text("endpoint_name",    "fraud-detection-endpoint", "Serving endpoint name")
dbutils.widgets.text("monitoring_schema","monitoring",               "Monitoring schema")
dbutils.widgets.text("inference_table",  "fraud_inference_payload",  "Inference table")
dbutils.widgets.text("workload_size",    "Small",                    "Workload size")
dbutils.widgets.text("scale_to_zero",    "true",                     "Scale to zero (true/false)")

catalog           = dbutils.widgets.get("catalog")
models_schema     = dbutils.widgets.get("models_schema")
model_name        = dbutils.widgets.get("model_name")
alias             = dbutils.widgets.get("alias")
endpoint_name     = dbutils.widgets.get("endpoint_name")
monitoring_schema = dbutils.widgets.get("monitoring_schema")
inference_table   = dbutils.widgets.get("inference_table")
workload_size     = dbutils.widgets.get("workload_size")
scale_to_zero     = dbutils.widgets.get("scale_to_zero").lower() == "true"

model_fqn = f"{catalog}.{models_schema}.{model_name}"

print(f"Model        : {model_fqn}@{alias}")
print(f"Endpoint     : {endpoint_name}")
print(f"Inference →  : {catalog}.{monitoring_schema}.{inference_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ensure monitoring schema exists

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{monitoring_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Resolve the Champion version (SDK needs a concrete version number)

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

champion_mv = mlflow_client.get_model_version_by_alias(model_fqn, alias)
champion_version = int(champion_mv.version)
print(f"Resolved {model_fqn}@{alias} → version {champion_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build the endpoint config via Databricks SDK

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
    AutoCaptureConfigInput,
)

w = WorkspaceClient()

served_entity = ServedEntityInput(
    entity_name=model_fqn,
    entity_version=str(champion_version),
    name="fraud-detector-champion",     # logical name within the endpoint
    workload_size=workload_size,
    scale_to_zero_enabled=scale_to_zero,
)

traffic = TrafficConfig(routes=[
    Route(served_model_name="fraud-detector-champion", traffic_percentage=100)
])

auto_capture = AutoCaptureConfigInput(
    catalog_name=catalog,
    schema_name=monitoring_schema,
    table_name_prefix=inference_table,   # final table becomes <prefix>_payload
    enabled=True,
)

endpoint_config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=[served_entity],
    traffic_config=traffic,
    auto_capture_config=auto_capture,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create or update the endpoint

# COMMAND ----------

existing = None
try:
    existing = w.serving_endpoints.get(name=endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists — updating its config.")
except Exception:
    print(f"Endpoint '{endpoint_name}' does not exist — creating it.")

if existing is None:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
    )
else:
    w.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=[served_entity],
        traffic_config=traffic,
        auto_capture_config=auto_capture,
    )

endpoint = w.serving_endpoints.get(name=endpoint_name)
print(f"Endpoint state: {endpoint.state.ready.value if endpoint.state and endpoint.state.ready else 'UNKNOWN'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Print the endpoint URL

# COMMAND ----------

workspace_url = spark.conf.get("spark.databricks.workspaceUrl", None)
if not workspace_url:
    workspace_url = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook().getContext().browserHostName().get()
    )

invocation_url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
ui_url         = f"https://{workspace_url}/ml/endpoints/{endpoint_name}"

print(f"Invocation URL : {invocation_url}")
print(f"UI URL         : {ui_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Smoke-test with a sample payload

# COMMAND ----------

# Build a minimal sample record that matches the feature schema. We look up one
# row from the feature table to guarantee column alignment. Any passthrough
# fields the model expects are supplied.
sample_pdf = (
    spark.table(f"{catalog}.gold.claim_features")
         .limit(1)
         .drop("feature_computed_at")
         .toPandas()
)

# The serving payload format expected by Databricks (Split orient).
payload = {
    "dataframe_split": {
        "columns": list(sample_pdf.columns),
        "data":    sample_pdf.astype(object).where(sample_pdf.notna(), None).values.tolist(),
    }
}

try:
    response = w.serving_endpoints.query(name=endpoint_name, dataframe_split=payload["dataframe_split"])
    print(f"Smoke test response: {response}")
except Exception as e:
    # Don't fail the whole task on smoke-test errors — surface and continue.
    print(f"Smoke test error (non-fatal): {e}")

# COMMAND ----------

dbutils.notebook.exit(
    f"endpoint={endpoint_name};version={champion_version};invocation_url={invocation_url}"
)
