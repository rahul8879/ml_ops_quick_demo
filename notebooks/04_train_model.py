# Databricks notebook source
# MAGIC %md # Train fraud detector

# COMMAND ----------

# MAGIC %pip install --quiet databricks-feature-engineering imbalanced-learn xgboost shap
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("catalog",         "wcqmlopsdemo",                         "catalog")
dbutils.widgets.text("silver_schema",   "silver",                               "silver schema")
dbutils.widgets.text("silver_table",    "clean_claims",                         "silver table")
dbutils.widgets.text("gold_schema",     "gold",                                 "gold schema")
dbutils.widgets.text("feature_table",   "claim_features",                       "feature table")
dbutils.widgets.text("models_schema",   "models",                               "models schema")
dbutils.widgets.text("model_name",      "fraud_detector",                       "model name")
dbutils.widgets.text("experiment_name", "/insurance_fraud/xgb_fraud_detection", "experiment")
dbutils.widgets.text("run_date",        datetime.utcnow().strftime("%Y-%m-%d"), "run date")

catalog         = dbutils.widgets.get("catalog")
silver_schema   = dbutils.widgets.get("silver_schema")
silver_table    = dbutils.widgets.get("silver_table")
gold_schema     = dbutils.widgets.get("gold_schema")
feature_table   = dbutils.widgets.get("feature_table")
models_schema   = dbutils.widgets.get("models_schema")
model_name      = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")
run_date        = dbutils.widgets.get("run_date")

silver_fqn  = f"{catalog}.{silver_schema}.{silver_table}"
feature_fqn = f"{catalog}.{gold_schema}.{feature_table}"
model_fqn   = f"{catalog}.{models_schema}.{model_name}"

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)
mlflow.autolog(log_input_examples=False, log_model_signatures=True, silent=True)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

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

training_df = training_set.load_df().toPandas()

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

y = training_df["FraudFound_P"].astype(int)
X = training_df.drop(columns=["FraudFound_P", "PolicyNumber"], errors="ignore")

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
X = X.fillna(0.0)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# COMMAND ----------

from imblearn.over_sampling import SMOTE

X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# COMMAND ----------

import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
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

    y_pred  = model.predict(X_holdout)
    y_proba = model.predict_proba(X_holdout)[:, 1]

    f1  = f1_score(y_holdout, y_pred)
    auc = roc_auc_score(y_holdout, y_proba)
    cm  = confusion_matrix(y_holdout, y_pred)

    mlflow.log_metric("holdout_f1",            f1)
    mlflow.log_metric("holdout_roc_auc",       auc)
    mlflow.log_metric("holdout_positive_rate", float(y_holdout.mean()))

    print(classification_report(y_holdout, y_pred, digits=4))

    prec, rec, _ = precision_recall_curve(y_holdout, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"PR (AUC≈{auc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    ax.set_title("Precision-Recall")
    fig.savefig("/tmp/pr_curve.png", bbox_inches="tight"); plt.close(fig)
    mlflow.log_artifact("/tmp/pr_curve.png", artifact_path="eval")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["legit", "fraud"]).plot(ax=ax)
    ax.set_title("Confusion matrix")
    fig.savefig("/tmp/confusion_matrix.png", bbox_inches="tight"); plt.close(fig)
    mlflow.log_artifact("/tmp/confusion_matrix.png", artifact_path="eval")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_sample = X_holdout.sample(min(1000, len(X_holdout)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, shap_sample, show=False)
        plt.savefig("/tmp/shap_summary.png", bbox_inches="tight"); plt.close()
        mlflow.log_artifact("/tmp/shap_summary.png", artifact_path="explainability")
    except Exception as e:
        print(f"shap skipped: {e}")

    try:
        fi = pd.DataFrame({
            "feature":    X_train_bal.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv", artifact_path="explainability")
    except Exception as e:
        print(f"fi csv skipped: {e}")

    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.xgboost,
        training_set=training_set,
        registered_model_name=model_fqn,
        input_example=X_holdout.head(5),
    )

    mlflow.set_tag("run_date",  run_date)
    mlflow.set_tag("use_case",  "insurance_claims_fraud")
    mlflow.set_tag("algorithm", "xgboost")

print(f"run_id={run_id}  model={model_fqn}  f1={f1:.4f}  auc={auc:.4f}")

# COMMAND ----------

dbutils.notebook.exit(f"run_id={run_id};holdout_f1={f1:.4f};holdout_roc_auc={auc:.4f}")
