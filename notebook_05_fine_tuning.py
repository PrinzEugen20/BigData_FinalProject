# notebook_05_fine_tuning.py
# Smart Solar Management — Week 4: Fine-Tuning & Regression
#
# Improvements over Week 3 baseline:
#   1. IRRADIATION + MODULE_TEMPERATURE added to regression (critical thermal derating)
#   2. DC_PER_GHI ratio feature engineered (panel conversion efficiency signal)
#   3. SOILING_PROXY re-added to classification (non-leaky soiling indicator)
#   4. MONTH_SIN/COS removed (0.0 importance — only 2 months of data)
#   5. GBTRegressor added as regression challenger
#   6. Expanded param grids for both RF and GBT classifiers
#
# Run: docker compose run --rm nb05

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings

warnings.filterwarnings("ignore")
FIGURES = "figures"
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Spark session
# ---------------------------------------------------------------------------
spark = (SparkSession.builder
         .appName("SmartSolar_05_FineTuning")
         .config("spark.sql.repl.eagerEval.enabled", True)
         # Increase driver/executor memory to prevent OOM on large CV grids
         .config("spark.driver.memory", "4g")
         .config("spark.executor.memory", "4g")
         # Reduce shuffle partitions — dataset fits in 20 partitions, 200 wastes RAM
         .config("spark.sql.shuffle.partitions", "20")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("WEEK 4 — FINE-TUNING & REGRESSION (IMPROVED)")
print("=" * 60)
df = spark.read.parquet("data/processed/solar_preprocessed.parquet")
print(f"\n  Total rows : {df.count():,}")
print(f"  Columns    : {df.columns}")

# ---------------------------------------------------------------------------
# 3. Daylight filter + feature engineering
# ---------------------------------------------------------------------------
df_day = df.filter(F.col("GHI") > 10)

df_day = (df_day
    .withColumn("HOUR",       F.hour("DATETIME").cast(DoubleType()))
    .withColumn("MONTH_NUM",  F.month("DATETIME").cast(DoubleType()))
    .withColumn("HOUR_SIN",   F.sin(F.col("HOUR")      * F.lit(2.0 * np.pi / 24.0)))
    .withColumn("HOUR_COS",   F.cos(F.col("HOUR")      * F.lit(2.0 * np.pi / 24.0)))
    # DC_PER_GHI: how much DC power the panel produces per unit of solar resource.
    # Drops when soiling or thermal derating occurs — strong cleaning signal.
    .withColumn("DC_PER_GHI",
        F.when(F.col("GHI") > 0, F.col("DC_POWER") / F.col("GHI")).otherwise(0.0))
    # TEMP_DIFF: difference between module and ambient temperature.
    # High delta means panels are running hot → efficiency loss.
    .withColumn("TEMP_DIFF",
        F.col("MODULE_TEMPERATURE") - F.col("AMBIENT_TEMPERATURE"))
)

print(f"\n  Daylight rows (GHI > 10): {df_day.count():,}")
print("  Engineered: DC_PER_GHI, TEMP_DIFF")

# ---------------------------------------------------------------------------
# 4. Feature definitions (evidence-based, trimmed dead weight)
# ---------------------------------------------------------------------------
# LEAKAGE-FREE feature set for CLEANING_FLAG classification:
# CLEANING_FLAG = 1 if SOILING_PROXY > 0.10 (line 211, notebook_02)
# Therefore: SOILING_PROXY, EFFICIENCY (=AC/DC), DC_PER_GHI (=DC/GHI) are ALL
# derived from DC_POWER which is the same signal SOILING_PROXY is computed from.
# Using any of them would let the model trivially reconstruct the label formula.
# Only weather sensor + astronomical + time features are leak-free.
CLF_FEATURES = [
    "GHI", "DNI", "DHI",
    "AIR_TEMPERATURE", "MODULE_TEMPERATURE",
    "WIND_SPEED", "SOLAR_ZENITH_ANGLE",
    "TEMP_DIFF", "HOUR_SIN", "HOUR_COS",
]
# Regression uses IRRADIATION (physical sensor) instead of GHI (satellite estimate)
# IRRADIATION is the strongest single predictor of DC_POWER (direct measurement)
REG_FEATURES = [
    "IRRADIATION", "GHI", "MODULE_TEMPERATURE", "AIR_TEMPERATURE",
    "SOLAR_ZENITH_ANGLE", "TEMP_DIFF", "HOUR_SIN", "HOUR_COS",
]
LABEL_CLF = "CLEANING_FLAG"
LABEL_REG = "DC_POWER"

print(f"\n  CLF features ({len(CLF_FEATURES)}): {CLF_FEATURES}")
print(f"  REG features ({len(REG_FEATURES)}): {REG_FEATURES}")

# ---------------------------------------------------------------------------
# 5. Train / test split (temporal: May → train, June → test)
# ---------------------------------------------------------------------------
train_df = df_day.filter(F.month("DATETIME") == 5)
test_df  = df_day.filter(F.month("DATETIME") == 6)
train_count = train_df.count()
test_count  = test_df.count()
print(f"\n  Train (May): {train_count:,}  |  Test (Jun): {test_count:,}")

# ---------------------------------------------------------------------------
# 6. Class weights (cost-sensitive, same logic as nb04)
# ---------------------------------------------------------------------------
n_neg = train_df.filter(F.col(LABEL_CLF) == 0).count()
n_pos = train_count - n_neg
ratio = n_neg / n_pos
print(f"  Class weight ratio: {ratio:.1f}:1")
train_df = train_df.withColumn(
    "sample_weight",
    F.when(F.col(LABEL_CLF) == 1, F.lit(float(ratio))).otherwise(F.lit(1.0))
)

# ---------------------------------------------------------------------------
# A. CLASSIFICATION FINE-TUNING
# ---------------------------------------------------------------------------
assembler_clf = VectorAssembler(inputCols=CLF_FEATURES, outputCol="features_raw", handleInvalid="skip")
scaler_clf    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
eval_auc      = BinaryClassificationEvaluator(labelCol=LABEL_CLF, metricName="areaUnderROC")
eval_f1       = MulticlassClassificationEvaluator(labelCol=LABEL_CLF, metricName="f1")

# --- A1. Random Forest ---
print("\n" + "-" * 50)
print("A1. Random Forest — Expanded Grid (10 features)")
print("-" * 50)
rf = RandomForestClassifier(
    featuresCol="features", labelCol=LABEL_CLF, weightCol="sample_weight",
    featureSubsetStrategy="sqrt", seed=42
)
rf_pipeline  = Pipeline(stages=[assembler_clf, scaler_clf, rf])
rf_grid      = (ParamGridBuilder()
    .addGrid(rf.maxDepth,  [10, 15, 20])
    .addGrid(rf.numTrees,  [100, 200])
    .build())
rf_cv = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=rf_grid,
                       evaluator=eval_auc, numFolds=3, seed=42, parallelism=2)
print(f"  Fitting {len(rf_grid)} combos × 3 folds...")
rf_cv_model = rf_cv.fit(train_df)
rf_best     = rf_cv_model.bestModel
rf_best_clf = rf_best.stages[-1]
print(f"  ✓ Best RF  numTrees={rf_best_clf.getNumTrees}  maxDepth={rf_best_clf.getMaxDepth()}")

rf_results = []
for params, score in zip(rf_grid, rf_cv_model.avgMetrics):
    rf_results.append({"maxDepth": params[rf.maxDepth], "numTrees": params[rf.numTrees], "AUC": score})
    print(f"    depth={params[rf.maxDepth]:2d}  trees={params[rf.numTrees]:3d}  AUC={score:.4f}")
rf_results_df = pd.DataFrame(rf_results)

# --- A2. GBT Classifier ---
print("\n" + "-" * 50)
print("A2. GBT Classifier — Expanded Grid")
print("-" * 50)
gbt = GBTClassifier(featuresCol="features", labelCol=LABEL_CLF, seed=42)
gbt_pipeline = Pipeline(stages=[assembler_clf, scaler_clf, gbt])
gbt_grid     = (ParamGridBuilder()
    .addGrid(gbt.maxIter,  [50, 100])
    .addGrid(gbt.stepSize, [0.05, 0.10])
    .addGrid(gbt.maxDepth, [4, 6])
    .build())
gbt_cv = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=gbt_grid,
                        evaluator=eval_auc, numFolds=3, seed=42, parallelism=2)
print(f"  Fitting {len(gbt_grid)} combos × 3 folds...")
gbt_cv_model = gbt_cv.fit(train_df)
gbt_best     = gbt_cv_model.bestModel
gbt_best_clf = gbt_best.stages[-1]
print(f"  ✓ Best GBT maxIter={gbt_best_clf.getMaxIter()}  stepSize={gbt_best_clf.getStepSize()}  maxDepth={gbt_best_clf.getMaxDepth()}")

gbt_results = []
for params, score in zip(gbt_grid, gbt_cv_model.avgMetrics):
    gbt_results.append({
        "maxIter": params[gbt.maxIter], "stepSize": params[gbt.stepSize],
        "maxDepth": params[gbt.maxDepth], "AUC": score
    })
    print(f"    iter={params[gbt.maxIter]:3d}  lr={params[gbt.stepSize]:.2f}  depth={params[gbt.maxDepth]}  AUC={score:.4f}")
gbt_results_df = pd.DataFrame(gbt_results)

# --- A3. Test evaluation ---
print("\n" + "-" * 50)
print("A3. Test-Set Evaluation")
print("-" * 50)
rf_preds  = rf_best.transform(test_df)
gbt_preds = gbt_best.transform(test_df)
rf_auc    = eval_auc.evaluate(rf_preds)
gbt_auc   = eval_auc.evaluate(gbt_preds)
rf_f1     = eval_f1.evaluate(rf_preds)
gbt_f1    = eval_f1.evaluate(gbt_preds)
print(f"  Random Forest  AUC={rf_auc:.4f}  F1={rf_f1:.4f}")
print(f"  GBT            AUC={gbt_auc:.4f}  F1={gbt_f1:.4f}")
best_clf_name = "GBT" if gbt_auc >= rf_auc else "Random Forest"

# Feature importances from best classifier
best_clf_model = gbt_best if gbt_auc >= rf_auc else rf_best
best_clf_stage = best_clf_model.stages[-1]
try:
    fi_clf = pd.DataFrame({
        "Feature": CLF_FEATURES,
        "Importance": best_clf_stage.featureImportances.toArray()
    }).sort_values("Importance", ascending=False)
    fi_clf.to_csv(f"{FIGURES}/model_05_clf_feature_importances.csv", index=False)
    print(f"\n  Top-3 CLF features ({best_clf_name}):")
    for _, r in fi_clf.head(3).iterrows():
        print(f"    {r['Feature']:25s}: {r['Importance']:.4f}")
except Exception as e:
    print(f"  Feature importance extract skipped: {e}")

# ---------------------------------------------------------------------------
# B. DC POWER REGRESSION (improved)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("B. DC POWER REGRESSION (IRRADIATION + MODULE_TEMP added)")
print("=" * 60)

assembler_reg = VectorAssembler(inputCols=REG_FEATURES, outputCol="features_raw", handleInvalid="skip")
scaler_reg    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
eval_rmse     = RegressionEvaluator(labelCol=LABEL_REG, predictionCol="prediction", metricName="rmse")
eval_r2       = RegressionEvaluator(labelCol=LABEL_REG, predictionCol="prediction", metricName="r2")

# RF Regressor
rfr = RandomForestRegressor(featuresCol="features", labelCol=LABEL_REG, seed=42)
rfr_pipeline = Pipeline(stages=[assembler_reg, scaler_reg, rfr])
rfr_grid     = (ParamGridBuilder()
    .addGrid(rfr.maxDepth, [10, 15])
    .addGrid(rfr.numTrees, [100, 200])
    .build())
# parallelism=1 (sequential) prevents JVM OOM after classification already consumed memory
rfr_cv = CrossValidator(estimator=rfr_pipeline, estimatorParamMaps=rfr_grid,
                        evaluator=eval_rmse, numFolds=3, seed=42, parallelism=1)

# GBT Regressor
gbtr = GBTRegressor(featuresCol="features", labelCol=LABEL_REG, seed=42)
gbtr_pipeline = Pipeline(stages=[assembler_reg, scaler_reg, gbtr])
gbtr_grid     = (ParamGridBuilder()
    .addGrid(gbtr.maxIter,  [50, 100])
    .addGrid(gbtr.stepSize, [0.05, 0.10])
    .addGrid(gbtr.maxDepth, [5, 8])
    .build())
gbtr_cv = CrossValidator(estimator=gbtr_pipeline, estimatorParamMaps=gbtr_grid,
                         evaluator=eval_rmse, numFolds=3, seed=42, parallelism=1)

print(f"\n  Fitting RF Regressor ({len(rfr_grid)} combos)...")
rfr_cv_model = rfr_cv.fit(train_df)
rfr_best     = rfr_cv_model.bestModel
rfr_preds    = rfr_best.transform(test_df)
rfr_rmse     = eval_rmse.evaluate(rfr_preds)
rfr_r2       = eval_r2.evaluate(rfr_preds)
print(f"  RF  Regressor  RMSE={rfr_rmse:.2f} W  R²={rfr_r2:.4f}")

print(f"\n  Fitting GBT Regressor ({len(gbtr_grid)} combos)...")
gbtr_cv_model = gbtr_cv.fit(train_df)
gbtr_best     = gbtr_cv_model.bestModel
gbtr_preds    = gbtr_best.transform(test_df)
gbtr_rmse     = eval_rmse.evaluate(gbtr_preds)
gbtr_r2       = eval_r2.evaluate(gbtr_preds)
print(f"  GBT Regressor  RMSE={gbtr_rmse:.2f} W  R²={gbtr_r2:.4f}")

best_reg_preds = gbtr_preds if gbtr_r2 >= rfr_r2 else rfr_preds
best_reg_rmse  = min(gbtr_rmse, rfr_rmse)
best_reg_r2    = max(gbtr_r2, rfr_r2)
best_reg_name  = "GBT Regressor" if gbtr_r2 >= rfr_r2 else "RF Regressor"
print(f"\n  ★ Best regressor: {best_reg_name}  RMSE={best_reg_rmse:.2f}  R²={best_reg_r2:.4f}")

# ---------------------------------------------------------------------------
# C. VISUALISATIONS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("C. GENERATING VISUALISATIONS")
print("=" * 60)
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# C1 — RF Tuning Heatmap
print("  Saving RF tuning heatmap...")
rf_pivot = rf_results_df.pivot(index="maxDepth", columns="numTrees", values="AUC")
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(rf_pivot, annot=True, fmt=".4f", cmap="YlGnBu", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Mean CV AUC-ROC (3-fold)"})
ax.set_title("Random Forest Hyperparameter Grid\n(AUC-ROC, 3-fold CV)", fontsize=13, weight="bold")
ax.set_xlabel("Number of Trees"); ax.set_ylabel("Max Depth")
plt.tight_layout(); fig.savefig(f"{FIGURES}/model_05a_rf_tuning_heatmap.png"); plt.close(fig)

# C2 — RF Learning Curve
print("  Saving RF learning curve...")
best_d  = rf_best_clf.getMaxDepth()
lc_data = rf_results_df[rf_results_df["maxDepth"] == best_d].sort_values("numTrees")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lc_data["numTrees"], lc_data["AUC"], marker="o", linewidth=2, color="#2196F3")
ax.fill_between(lc_data["numTrees"], lc_data["AUC"] - 0.005, lc_data["AUC"] + 0.005,
                alpha=0.2, color="#2196F3", label="±0.005 band")
ax.set_title(f"RF Learning Curve (maxDepth={best_d})\nAUC-ROC vs Number of Trees", fontsize=13, weight="bold")
ax.set_xlabel("Number of Trees"); ax.set_ylabel("AUC-ROC (3-fold CV)")
ax.set_ylim(max(0, lc_data["AUC"].min() - 0.02), min(1.0, lc_data["AUC"].max() + 0.02))
ax.legend(); plt.tight_layout(); fig.savefig(f"{FIGURES}/model_05b_rf_learning_curve.png"); plt.close(fig)

# C3 — GBT Tuning Curve
print("  Saving GBT tuning curve...")
fig, ax = plt.subplots(figsize=(7, 4))
colors = {0.05: "#E91E63", 0.10: "#FF9800"}
for lr, grp in gbt_results_df.groupby("stepSize"):
    best_d_grp = grp.groupby("maxIter")["AUC"].max().reset_index()
    ax.plot(best_d_grp["maxIter"], best_d_grp["AUC"], marker="s", linewidth=2,
            color=colors.get(lr, "grey"), label=f"stepSize={lr}")
ax.set_title("GBT Tuning Curve\nAUC-ROC vs Boosting Iterations", fontsize=13, weight="bold")
ax.set_xlabel("Max Iterations"); ax.set_ylabel("AUC-ROC (3-fold CV)")
ax.legend(title="Learning Rate"); plt.tight_layout()
fig.savefig(f"{FIGURES}/model_05c_gbt_tuning_curve.png"); plt.close(fig)

# C4 — RF vs GBT comparison bar
print("  Saving RF vs GBT comparison...")
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, [rf_auc, rf_f1],  w, label="Random Forest", color="#2196F3")
b2 = ax.bar(x + w/2, [gbt_auc, gbt_f1], w, label="GBT",          color="#E91E63")
ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=9)
ax.set_title("Algorithm Comparison — Test Set (June 2020)", fontsize=13, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(["AUC-ROC", "F1-Score"])
ax.set_ylabel("Score"); ax.set_ylim(0, 1.1); ax.legend()
plt.tight_layout(); fig.savefig(f"{FIGURES}/model_05d_rf_vs_gbt.png"); plt.close(fig)

# C5 — CLF feature importances
print("  Saving CLF feature importance chart...")
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    fi_plot = fi_clf.sort_values("Importance")
    ax.barh(fi_plot["Feature"], fi_plot["Importance"], color="#4CAF50")
    ax.set_title(f"Feature Importances — {best_clf_name} Classifier\n(Gini Impurity Reduction)",
                 fontsize=13, weight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout(); fig.savefig(f"{FIGURES}/model_05e_clf_feature_importance.png"); plt.close(fig)
except Exception:
    pass

# C6 — Regression Predicted vs Actual
print("  Saving regression predicted vs actual...")
reg_sample = (best_reg_preds.orderBy("DATETIME")
    .select("DATETIME", LABEL_REG, "prediction").limit(500).toPandas())
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(reg_sample.index, reg_sample[LABEL_REG],    label="Actual DC Power",    lw=1.5, color="#333")
ax.plot(reg_sample.index, reg_sample["prediction"], label="Predicted DC Power", lw=1.5, color="#2196F3", ls="--")
ax.set_title(f"Regression — Predicted vs Actual DC Power\n({best_reg_name} | RMSE={best_reg_rmse:.1f} W, R²={best_reg_r2:.4f})",
             fontsize=13, weight="bold")
ax.set_xlabel("Sample Index (chronological)"); ax.set_ylabel("DC Power (W)"); ax.legend()
plt.tight_layout(); fig.savefig(f"{FIGURES}/model_05f_regression_pred_vs_actual.png"); plt.close(fig)

# C7 — Regression: RF vs GBT RMSE comparison
print("  Saving regressor comparison...")
fig, ax = plt.subplots(figsize=(6, 4))
models  = ["RF Regressor", "GBT Regressor"]
rmses   = [rfr_rmse, gbtr_rmse]
r2s     = [rfr_r2, gbtr_r2]
x = np.arange(2); w = 0.35
b1 = ax.bar(x - w/2, rmses, w, label="RMSE (W)", color="#FF7043")
b2 = ax.bar(x + w/2, [r * 1000 for r in r2s], w, label="R² ×1000", color="#26A69A")
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=9)
ax.bar_label(b2, labels=[f"{r:.4f}" for r in r2s], padding=3, fontsize=9)
ax.set_title("Regressor Comparison — RMSE & R²", fontsize=13, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(models); ax.legend()
plt.tight_layout(); fig.savefig(f"{FIGURES}/model_05g_regressor_comparison.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("WEEK 4 FINAL SUMMARY")
print("=" * 60)
print(f"\n  CLASSIFICATION (CLEANING_FLAG)")
print(f"  {'Algorithm':20s} {'CV AUC':>8s} {'Test AUC':>9s} {'Test F1':>8s}")
print(f"  {'-'*50}")
best_rf_cv  = max(r["AUC"] for r in rf_results)
best_gbt_cv = max(r["AUC"] for r in gbt_results)
print(f"  {'Random Forest':20s} {best_rf_cv:>8.4f} {rf_auc:>9.4f} {rf_f1:>8.4f}")
print(f"  {'GBT':20s} {best_gbt_cv:>8.4f} {gbt_auc:>9.4f} {gbt_f1:>8.4f}")
print(f"\n  ★ Best classifier : {best_clf_name}")

print(f"\n  REGRESSION (DC_POWER)")
print(f"  {'Algorithm':20s} {'RMSE (W)':>10s} {'R²':>8s}")
print(f"  {'-'*40}")
print(f"  {'RF Regressor':20s} {rfr_rmse:>10.2f} {rfr_r2:>8.4f}")
print(f"  {'GBT Regressor':20s} {gbtr_rmse:>10.2f} {gbtr_r2:>8.4f}")
print(f"\n  ★ Best regressor  : {best_reg_name}")
print(f"    RMSE = {best_reg_rmse:.2f} W   R² = {best_reg_r2:.4f}")

print(f"\n  FIGURES (7 total):")
for f in sorted(os.listdir(FIGURES)):
    if f.startswith("model_05"):
        print(f"    {FIGURES}/{f}")

print("\nNotebook 05 complete.")
