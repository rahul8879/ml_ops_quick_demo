# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Train Fraud Detection Model (XGBoost + SMOTE + MLflow)
# MAGIC
# MAGIC **Purpose:** Train an `XGBClassifier` on the claim features. Handle ~6% class
# MAGIC imbalance with SMOTE, track everything to MLflow, log a SHAP summary plot, and
# MAGIC register the resulting model to Unity Catalog via `FeatureEngineeringClient.log_model`.
# MAGIC
# MAGIC **Inputs**
# MAGIC - Feature table: `insurance_demo.gold.claim_features`
# MAGIC - Label source : `insurance_demo.silver.clean_claims` (column `FraudFound_P`)
# MAGIC
# MAGIC **Outputs**
# MAGIC - MLflow experiment: `/insurance_fraud/xgb_fraud_detection`
# MAGIC - UC model: `insurance_demo.models.fraud_detector` (new version registered)
# MAGIC
# MAGIC **Runtime:** Databricks Runtime 15.4 LTS ML

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering imbalanced-learn xgboost shap
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widgets

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",         "insurance_demo",                       "Unity Catalog")
dbutils.widgets.text("silver_schema",   "silver",                               "Silver schema")
dbutils.widgets.text("silver_table",    "clean_claims",                         "Silver table (labels)")
dbutils.widgets.text("gold_schema",     "gold",                                 "Gold schema")
dbutils.widgets.text("feature_table",   "claim_features",                       "Feature table")
dbutils.widgets.text("models_schema",   "models",                               "Models schema")
dbutils.widgets.text("model_name",      "fraud_detector",                       "Model name")
dbutils.widgets.text("experiment_name", "/insurance_fraud/xgb_fraud_detection", "MLflow experiment path")
dbutils.widgets.text("run_date",        datetime.utcnow().strftime("%Y-%m-%d"), "Run date")

catalog         = dbutils.widgets.get("catalog")
silver_schema   = dbutils.widgets.get("silver_schema")
silver_table    = dbutils.widgets.get("silver_table")
gold_schema     = dbutils.widgets.get("gold_schema")
feature_table   = dbutils.widgets.get("feature_table")
models_schema   = dbutils.widgets.get("models_schema")
model_name      = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
run_date        = dbutils.widgets.get("run_date")

silver_fqn   = f"{catalog}.{silver_schema}.{silver_table}"
feature_fqn  = f"{catalog}.{gold_schema}.{feature_table}"
model_fqn    = f"{catalog}.{models_schema}.{model_name}"

print(f"Feature table : {feature_fqn}")
print(f"Label source  : {silver_fqn}")
print(f"Model (UC)    : {model_fqn}")
print(f"Experiment    : {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure MLflow for Unity Catalog + autolog

# COMMAND ----------

import mlflow

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{models_schema}")

mlflow.set_registry_uri("databricks-uc")            # UC-backed registry
mlflow.set_experiment(experiment_name)
mlflow.autolog(log_input_examples=False, log_model_signatures=True, silent=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build training set with FeatureLookup

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# The "ground truth" dataframe has keys + labels + (optionally) exclude-from-lookup columns.
# We keep only PolicyNumber + FraudFound_P here so FeatureLookup fills the rest.
ground_truth_df = (
    spark.table(silver_fqn)
         .select("PolicyNumber", "FraudFound_P")
         .dropna(subset=["PolicyNumber", "FraudFound_P"])
)

# List the feature columns we actually want from the feature table.
feature_cols_to_lookup = [
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

# Filter to columns that actually exist in the feature table (robustness).
feat_schema_cols = [c.name for c in spark.table(feature_fqn).schema.fields]
feature_cols_to_lookup = [c for c in feature_cols_to_lookup if c in feat_schema_cols]

feature_lookups = [
    FeatureLookup(
        table_name=feature_fqn,
        lookup_key="PolicyNumber",
        feature_names=feature_cols_to_lookup,
    )
]

training_set = fe.create_training_set(
    df=ground_truth_df,
    feature_lookups=feature_lookups,
    label="FraudFound_P",
    exclude_columns=[],
)

training_df = training_set.load_df().toPandas()
print(f"Training set shape: {training_df.shape}")
print(f"Label balance:\n{training_df['FraudFound_P'].value_counts(normalize=True)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train / validation split

# COMMAND ----------

from sklearn.model_selection import train_test_split

y = training_df["FraudFound_P"].astype(int)
X = training_df.drop(columns=["FraudFound_P", "PolicyNumber"], errors="ignore")

# Any remaining categoricals → one-hot encode (driver_age_bucket in particular).
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
if cat_cols:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    X = __import__("pandas").get_dummies(X, columns=cat_cols, drop_first=False)

# Impute any remaining NaNs with 0 (tree models handle this fine, but SMOTE can't).
X = X.fillna(0.0)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train shape:   {X_train.shape}, positives={y_train.sum()}")
print(f"Holdout shape: {X_holdout.shape}, positives={y_holdout.sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. SMOTE on the training fold only
# MAGIC
# MAGIC SMOTE is applied **after** the train/holdout split and only to the training
# MAGIC fold — oversampling the holdout would leak into the metrics.

# COMMAND ----------

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"Post-SMOTE train shape: {X_train_bal.shape}")
print(f"Post-SMOTE label mix  : {y_train_bal.value_counts(normalize=True).to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train XGBoost + log everything to MLflow

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

with mlflow.start_run(run_name=f"xgb_fraud_{run_date}") as run:
    run_id = run.info.run_id

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_bal, y_train_bal)

    # ---- Holdout evaluation --------------------------------------------
    y_pred  = model.predict(X_holdout)
    y_proba = model.predict_proba(X_holdout)[:, 1]

    f1  = f1_score(y_holdout, y_pred)
    auc = roc_auc_score(y_holdout, y_proba)
    cm  = confusion_matrix(y_holdout, y_pred)

    mlflow.log_metric("holdout_f1",       f1)
    mlflow.log_metric("holdout_roc_auc",  auc)
    mlflow.log_metric("holdout_positive_rate", float(y_holdout.mean()))

    print(classification_report(y_holdout, y_pred, digits=4))
    print(f"F1={f1:.4f}   ROC-AUC={auc:.4f}")

    # ---- Precision-Recall curve plot ------------------------------------
    prec, rec, _ = precision_recall_curve(y_holdout, y_proba)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.plot(rec, prec, label=f"PR curve (AUC≈{auc:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall — fraud class")
    ax_pr.legend()
    pr_path = "/tmp/pr_curve.png"
    fig_pr.savefig(pr_path, bbox_inches="tight")
    plt.close(fig_pr)
    mlflow.log_artifact(pr_path, artifact_path="eval")

    # ---- Confusion matrix plot ------------------------------------------
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["legit", "fraud"]).plot(ax=ax_cm)
    ax_cm.set_title("Confusion matrix — holdout")
    cm_path = "/tmp/confusion_matrix.png"
    fig_cm.savefig(cm_path, bbox_inches="tight")
    plt.close(fig_cm)
    mlflow.log_artifact(cm_path, artifact_path="eval")

    # ---- SHAP summary plot ---------------------------------------------
    try:
        explainer = shap.TreeExplainer(model)
        # Use a subsample for speed; SHAP handles the full matrix but it's slow.
        shap_sample = X_holdout.sample(min(1000, len(X_holdout)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, shap_sample, show=False)
        shap_path = "/tmp/shap_summary.png"
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(shap_path, artifact_path="explainability")
    except Exception as e:
        # Never fail the run just because explainability plotting blew up.
        print(f"SHAP plot failed (non-fatal): {e}")

    # ---- Log feature importance table as an artifact --------------------
    try:
        import pandas as pd
        fi = pd.DataFrame({
            "feature":    X_train_bal.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_path = "/tmp/feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path, artifact_path="explainability")
    except Exception as e:
        print(f"Feature importance logging failed: {e}")

    # ---- Register model to Unity Catalog via Feature Engineering --------
    # fe.log_model binds the model to the training set / feature lookups so
    # batch scoring and serving will automatically re-join features.
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.xgboost,
        training_set=training_set,
        registered_model_name=model_fqn,
        input_example=X_holdout.head(5),
    )

    mlflow.set_tag("run_date",   run_date)
    mlflow.set_tag("use_case",   "insurance_claims_fraud")
    mlflow.set_tag("algorithm",  "xgboost")

print(f"\nMLflow run_id       : {run_id}")
print(f"Registered UC model : {model_fqn}")

# COMMAND ----------

dbutils.notebook.exit(
    f"run_id={run_id};holdout_f1={f1:.4f};holdout_roc_auc={auc:.4f}"
)
