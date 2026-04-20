# Vehicle Insurance Claims — Fraud Detection MLOps on Databricks

End-to-end MLOps pipeline for the Kaggle *Vehicle Claim Fraud Detection* dataset (`shivamb`, 15k rows, 33 features, target `FraudFound_P`). Built for Azure Databricks with Unity Catalog, Feature Engineering, MLflow UC-registry aliases, Mosaic AI Model Serving, and Lakehouse Monitoring — wired together as a single multi-task workflow via a Databricks Asset Bundle, and shipped through a GitHub Actions pipeline.

## Architecture at a glance

```
Raw CSV (UC Volume)
        │   Auto Loader (cloudFiles)
        ▼
insurance_demo.bronze.raw_claims  ── 01_ingest_bronze
        │   cast + dedup + MERGE + CDF
        ▼
insurance_demo.silver.clean_claims  ── 02_transform_silver
        │   feature engineering
        ▼
insurance_demo.gold.claim_features  ── 03_feature_engineering  (FeatureEngineeringClient)
        │   FeatureLookup + SMOTE + XGBoost + MLflow autolog + SHAP
        ▼
insurance_demo.models.fraud_detector  ── 04_train_model  (registered to UC)
        │   gates (F1 ≥ 0.72, AUC ≥ 0.80, PSI ≤ 0.20) + alias swap
        ▼
@Challenger  →  @Champion   ── 05_validate_promote
        │   Databricks SDK (WorkspaceClient)
        ▼
Mosaic AI Serving endpoint: fraud-detection-endpoint  ── 06_deploy_serving
        │   inference table autocapture
        ▼
insurance_demo.monitoring.fraud_inference_payload
        │   quality_monitors.create (InferenceLog, CLASSIFICATION, hourly)
        ▼
Lakehouse Monitor + auto dashboard  ── 07_monitoring
```

## File structure

```
.
├── databricks.yml                     # Asset Bundle — wires the 7 notebooks as a single Job
├── README.md                          # This file
├── .github/workflows/deploy.yml       # CI: validate on PR, deploy on push to main
└── notebooks/
    ├── 01_ingest_bronze.py            # Auto Loader CSV → Delta bronze
    ├── 02_transform_silver.py         # Cast + dedup + MERGE + CDF
    ├── 03_feature_engineering.py      # FeatureEngineeringClient → gold feature table
    ├── 04_train_model.py              # SMOTE + XGBoost + MLflow + SHAP + fe.log_model
    ├── 05_validate_promote.py         # F1/AUC/PSI gates + UC Challenger/Champion aliases
    ├── 06_deploy_serving.py           # Mosaic AI endpoint via Databricks SDK
    └── 07_monitoring.py               # Lakehouse Monitor on inference table
```

## Unity Catalog namespace

| Layer | Table | Purpose |
|---|---|---|
| Bronze | `insurance_demo.bronze.raw_claims` | Append-only raw ingest |
| Silver | `insurance_demo.silver.clean_claims` | Clean + dedup (MERGE, CDF) |
| Gold / Features | `insurance_demo.gold.claim_features` | Feature Store table (PK = `PolicyNumber`) |
| Model | `insurance_demo.models.fraud_detector` | UC-registered model, alias `Champion` |
| Monitoring | `insurance_demo.monitoring.fraud_inference_payload` | Inference capture (serving) |
| Monitoring | `insurance_demo.monitoring.model_validation_log` | Validation audit log |

## Technical highlights

- **Feature Store**: uses `databricks.feature_engineering.FeatureEngineeringClient` (the current API). The deprecated `FeatureStoreClient` is avoided everywhere.
- **MLflow UC registry**: `mlflow.set_registry_uri("databricks-uc")` and `fe.log_model(...)` bind the model to its feature-lookup metadata for online/offline scoring.
- **No MLflow stages**: promotion is done with UC aliases (`Challenger`, `Champion`) via `MlflowClient.set_registered_model_alias`.
- **Class imbalance**: SMOTE is applied only to the training fold after a stratified split so holdout metrics remain honest.
- **Drift gate**: PSI vs. the training baseline; any feature with PSI > 0.2 fails the promotion gate.
- **Serving**: created/updated through `WorkspaceClient.serving_endpoints` (no raw REST). `scale_to_zero_enabled=True`, `workload_size="Small"`, 100% traffic to Champion, inference auto-capture enabled.
- **Monitoring**: `WorkspaceClient.quality_monitors.create` with `MonitorInferenceLog` + `PROBLEM_TYPE_CLASSIFICATION`, baseline = feature table, hourly cron, auto dashboard URL printed on completion.
- **Runtime**: all clusters target **Databricks Runtime 15.4 LTS ML**.

## Deploy from scratch

### 1. One-time prerequisites

- A Unity-Catalog enabled Azure Databricks workspace.
- A UC Volume holding the raw CSVs — the bronze notebook defaults to `/Volumes/insurance_demo/raw/claims/`. Upload the Kaggle file there.
- The catalog `insurance_demo` (the bundle will create schemas on first run, but you need the catalog to exist or grant `CREATE CATALOG` to the deploying principal).
- A PAT or service-principal token stored in GitHub Secrets as `DATABRICKS_TOKEN`, and the workspace URL as `DATABRICKS_HOST`.

### 2. Local validation

```bash
# Install the Databricks CLI (bundles require v0.205+)
brew tap databricks/tap && brew install databricks

# From the repo root
databricks bundle validate --target dev
databricks bundle validate --target prod
```

### 3. Deploy

```bash
# Deploy to dev (schedule is PAUSED by default)
databricks bundle deploy --target dev

# Trigger a one-off run
databricks bundle run insurance_fraud_pipeline --target dev

# Promote to prod (schedule is UNPAUSED)
databricks bundle deploy --target prod
```

### 4. CI/CD (GitHub Actions)

- On pull request → `databricks bundle validate` runs for both targets.
- On push to `main` → `databricks bundle deploy --target prod` runs.

The workflow is at `.github/workflows/deploy.yml`.

## Configuration

All notebooks expose widgets (`dbutils.widgets`) so they can be rebound per environment via `base_parameters` in `databricks.yml`. The two most commonly overridden values are `catalog` and `run_date`. The serving endpoint name and notification email are variables on the bundle (`${var.endpoint_name}`, `${var.notifications_email}`).

## Schedule

Daily at **02:00 Asia/Kolkata** on the prod target (`quartz_cron_expression: "0 0 2 * * ?"`, `timezone_id: "Asia/Kolkata"`). The dev target keeps the schedule paused so it only runs on demand.

## Troubleshooting

- **`No versions found for insurance_demo.models.fraud_detector`** — run task `train_model` at least once before `validate_promote`.
- **Feature table registration fails** — confirm the deploying principal has `USE CATALOG` / `CREATE TABLE` on `insurance_demo.gold`.
- **Serving endpoint stuck in `UPDATING`** — check the endpoint event log in Databricks UI; most common causes are insufficient workspace admin entitlements or the model artifact missing its `input_example` / signature.
- **Lakehouse Monitor dashboard empty** — the inference table needs rows before the first refresh; hit the endpoint with a sample payload (notebook 06 does this automatically).
