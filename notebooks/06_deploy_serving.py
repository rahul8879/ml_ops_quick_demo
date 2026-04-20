# Databricks notebook source
# MAGIC %md # Deploy serving endpoint

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk>=0.30.0 mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog",           "wcqmlopsdemo",              "catalog")
dbutils.widgets.text("models_schema",     "models",                    "models schema")
dbutils.widgets.text("model_name",        "fraud_detector",            "model name")
dbutils.widgets.text("alias",             "Champion",                  "alias")
dbutils.widgets.text("endpoint_name",     "fraud-detection-endpoint",  "endpoint")
dbutils.widgets.text("monitoring_schema", "monitoring",                "monitoring schema")
dbutils.widgets.text("inference_table",   "fraud_inference_payload",   "inference table prefix")
dbutils.widgets.text("workload_size",     "Small",                     "workload size")
dbutils.widgets.text("scale_to_zero",     "true",                      "scale to zero")

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

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    champion_version = int(client.get_model_version_by_alias(model_fqn, alias).version)
except Exception:
    versions = client.search_model_versions(f"name = '{model_fqn}'")
    assert versions, f"no versions for {model_fqn}"
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    champion_version = int(latest.version)
    client.set_registered_model_alias(name=model_fqn, alias=alias, version=champion_version)
    print(f"no {alias} alias — seeded on latest v{champion_version}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
)

w = WorkspaceClient()

served_entity = ServedEntityInput(
    entity_name=model_fqn,
    entity_version=str(champion_version),
    name="fraud-detector-champion",
    workload_size=workload_size,
    scale_to_zero_enabled=scale_to_zero,
)

traffic = TrafficConfig(routes=[
    Route(served_model_name="fraud-detector-champion", traffic_percentage=100),
])

ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=catalog,
        schema_name=monitoring_schema,
        table_name_prefix=inference_table,
    ),
)

endpoint_exists = False
try:
    w.serving_endpoints.get(name=endpoint_name)
    endpoint_exists = True
except Exception:
    endpoint_exists = False

if endpoint_exists:
    w.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=[served_entity],
        traffic_config=traffic,
    )
    w.serving_endpoints.put_ai_gateway(
        name=endpoint_name,
        inference_table_config=ai_gateway.inference_table_config,
    )
else:
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=[served_entity],
            traffic_config=traffic,
        ),
        ai_gateway=ai_gateway,
    )

# COMMAND ----------

workspace_url = spark.conf.get("spark.databricks.workspaceUrl", None) \
    or dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()

invocation_url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
ui_url         = f"https://{workspace_url}/ml/endpoints/{endpoint_name}"

print(f"invocation: {invocation_url}")
print(f"ui:         {ui_url}")

# COMMAND ----------

sample_pdf = (
    spark.table(f"{catalog}.gold.claim_features")
         .limit(1)
         .drop("feature_computed_at", "PolicyNumber")
         .toPandas()
         .astype(float)
         .fillna(0.0)
)

try:
    response = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_records=sample_pdf.to_dict(orient="records"),
    )
    print(response)
except Exception as e:
    print(f"smoke test: {e}")

# COMMAND ----------

dbutils.notebook.exit(
    f"endpoint={endpoint_name};version={champion_version};invocation_url={invocation_url}"
)
