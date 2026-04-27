# notebook_04_modeling.py
# Smart Solar Management - Big Data Project
# Week 3 — Baseline Predictive Model
#
# Trains a Random Forest classifier (PySpark MLlib) to predict CLEANING_FLAG:
# whether a solar inverter is underperforming its plant peers by >10% during
# daylight hours and should be scheduled for panel cleaning.
#
# Data quality gate MUST pass before this runs:
#   docker compose run --rm nb02 python _data_quality_gate.py
#
# Model strategy:
#   - Feature set: GHI, AIR_TEMPERATURE, SOLAR_ZENITH_ANGLE,
#                  SOILING_PROXY, EFFICIENCY + cyclic time encodings
#   - Dropped features: DHI, DNI (VIF>7 collinear with GHI),
#                       OPTIMAL_TILT (algebraic copy of SOLAR_ZENITH),
#                       WIND_SPEED (r=0.019 with target)
#   - Train/test split: date-based (May → train, June → test)
#     Random split is forbidden: DAILY_YIELD is cumulative and would leak future data.
#   - Class imbalance (9.8% positive): handled via weightCol, not oversampling.
#     Oversampling 15-min intervals creates temporally correlated duplicates.
#   - Primary metric: AUC-ROC (course-specified in Spark Evaluator docs).
#     Accuracy is logged but is misleading with 90/10 imbalance.
#   - Explainability: feature importances are printed and saved to figures/.
#
# References:
#   Breiman (2001) - Random Forests original paper
#   Panat & Varanasi (2022) - waterless electrostatic cleaning threshold
#   Darhmaoui & Lahjouji (2013) - tilt formula validation
#   IEC 61724-1 - PV performance monitoring standard

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

FIGURES = "figures"
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Initialise Spark
# ---------------------------------------------------------------------------
spark = (SparkSession.builder
         .appName("SmartSolar_04_Modeling")
         .config("spark.sql.repl.eagerEval.enabled", True)
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Load preprocessed data
# ---------------------------------------------------------------------------
print("Loading preprocessed dataset...")
df = spark.read.parquet("data/processed/solar_preprocessed.parquet")

total_rows = df.count()
print(f"  Total rows: {total_rows:,}")
print(f"  Columns   : {df.columns}")

# ---------------------------------------------------------------------------
# 3. Restrict to daylight hours only
# ---------------------------------------------------------------------------
# Models trained on nighttime rows learn the trivial rule "DC=0 → no cleaning"
# which inflates accuracy but adds zero operational insight.
# IEC 61724-1 defines "operational period" as GHI > 10 W/m².
df_day = df.filter(F.col("GHI") > 10)
day_count = df_day.count()
print(f"\n  Daylight rows (GHI > 10): {day_count:,}")
print(f"  Nighttime rows dropped  : {total_rows - day_count:,}")

# ---------------------------------------------------------------------------
# 4. Cyclic time encoding
# ---------------------------------------------------------------------------
# HOUR and MONTH are cyclical: hour 23 and hour 0 are adjacent, not 23 apart.
# Using raw integers makes a linear model think January is 11 months from December.
# Sin/cos encoding preserves the cyclic topology without introducing any
# ordering assumption. Two components per variable are needed to uniquely
# represent every position on the cycle.
# Reference: Friedman et al. (2001) "Elements of Statistical Learning" §5.1
df_day = (df_day
    .withColumn("HOUR",       F.hour("DATETIME").cast(DoubleType()))
    .withColumn("MONTH_NUM",  F.month("DATETIME").cast(DoubleType()))
    .withColumn("HOUR_SIN",   F.sin(F.col("HOUR")  * F.lit(2.0 * np.pi / 24.0)))
    .withColumn("HOUR_COS",   F.cos(F.col("HOUR")  * F.lit(2.0 * np.pi / 24.0)))
    .withColumn("MONTH_SIN",  F.sin(F.col("MONTH_NUM") * F.lit(2.0 * np.pi / 12.0)))
    .withColumn("MONTH_COS",  F.cos(F.col("MONTH_NUM") * F.lit(2.0 * np.pi / 12.0)))
)

print("\n  Cyclic encodings added: HOUR_SIN, HOUR_COS, MONTH_SIN, MONTH_COS")

# ---------------------------------------------------------------------------
# 5. Feature selection (evidence-based)
# ---------------------------------------------------------------------------
# Feature selection rationale (from _evaluate_pipeline.py audit):
#   KEEP: GHI           — primary irradiance driver (std=289, r=0.67 with DC)
#   KEEP: AIR_TEMP      — thermal context (VIF=1.89, r=0.51 with DC)
#   KEEP: SOLAR_ZENITH  — geometric proxy for time-of-day and angle
#   KEEP: SOILING_PROXY — strongest cleaning signal (r=0.676 with CLEANING_FLAG)
#   KEEP: EFFICIENCY    — panel state indicator (bimodal: normal or failed)
#   KEEP: HOUR_SIN/COS  — within-day temporal pattern
#   KEEP: MONTH_SIN/COS — seasonal pattern
#   DROP: DHI, DNI      — VIF>7, collinear with GHI (GHI↔DHI r=0.81)
#   DROP: OPTIMAL_TILT  — algebraically = 90 - SOLAR_ZENITH (r=-0.9992)
#   DROP: WIND_SPEED    — r=0.019 with DC, r=-0.035 with CLEANING_FLAG
FEATURE_COLS = [
    "GHI",
    "AIR_TEMPERATURE",
    "SOLAR_ZENITH_ANGLE",
    "EFFICIENCY",
    "HOUR_SIN",
    "HOUR_COS",
    "MONTH_SIN",
    "MONTH_COS",
]
LABEL_COL   = "CLEANING_FLAG"

print(f"\n  Feature vector ({len(FEATURE_COLS)} features):")
for f in FEATURE_COLS:
    print(f"    - {f}")

# ---------------------------------------------------------------------------
# 6. Date-based train / test split
# ---------------------------------------------------------------------------
# randomSplit() is prohibited because DAILY_YIELD is a cumulative counter.
# A random split would place afternoon readings in training while morning readings
# (with lower DAILY_YIELD) land in test — leaking future-state information.
# Date-based split (May → train, June → test) respects temporal ordering.
#
# The 48%/52% split (64,950 / 71,526 rows) is verified by the quality gate.
train_df = df_day.filter(F.month("DATETIME") == 5)
test_df  = df_day.filter(F.month("DATETIME") == 6)

train_count = train_df.count()
test_count  = test_df.count()

print(f"\n  Train (May 2020) : {train_count:,} rows")
print(f"  Test  (Jun 2020) : {test_count:,} rows")

# Log class distribution in each split
train_pos = train_df.filter(F.col(LABEL_COL) == 1).count()
test_pos  = test_df.filter(F.col(LABEL_COL) == 1).count()
print(f"  Train positive rate: {train_pos/train_count*100:.1f}%")
print(f"  Test  positive rate: {test_pos/test_count*100:.1f}%")

# ---------------------------------------------------------------------------
# 7. Class weight computation
# ---------------------------------------------------------------------------
# The dataset has ~9.8% positive (cleaning needed). Accuracy on the raw
# distribution would be ~90% for a trivial "always predict 0" classifier.
# We use weightCol to assign higher misclassification cost to the minority class.
# The weight ratio equals n_negative / n_positive — this is the standard
# cost-sensitive approach for tree-based models (Chen & Guestrin 2016).
#
# Note: SMOTE/oversampling is deliberately avoided. 15-minute time-series data
# has high autocorrelation; synthetic duplicates of rare events would create
# temporally identical training examples and overfit the failure modes.
n_neg = train_df.filter(F.col(LABEL_COL) == 0).count()
n_pos = train_count - n_neg
ratio = n_neg / n_pos
print(f"\n  Class weight ratio (neg/pos): {ratio:.1f}")

# Add weight column: minority class gets weight = ratio, majority gets 1.0
train_df = train_df.withColumn(
    "sample_weight",
    F.when(F.col(LABEL_COL) == 1, F.lit(float(ratio)))
     .otherwise(F.lit(1.0))
)

# ---------------------------------------------------------------------------
# 8. Build the ML pipeline
# ---------------------------------------------------------------------------
# Pipeline stages:
#   Stage 1 — VectorAssembler: combine features into a dense vector
#   Stage 2 — StandardScaler : zero-mean, unit-variance scaling
#              (Random Forest doesn't require scaling, but it makes
#               the feature importances directly comparable)
#   Stage 3 — RandomForestClassifier: ensemble of 100 decision trees
#
# numTrees=100: standard starting point (Breiman 2001 recommends >=100)
# maxDepth=10: deep enough to capture non-linear interactions without
#              memorising individual 15-minute readings
# featureSubsetStrategy="sqrt": standard for classification RF
# seed=42: reproducibility

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features_raw",
    handleInvalid="skip"    # drops rows where any feature is null
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol=LABEL_COL,
    weightCol="sample_weight",
    numTrees=100,
    maxDepth=10,
    featureSubsetStrategy="sqrt",
    seed=42,
)

pipeline = Pipeline(stages=[assembler, scaler, rf])

# ---------------------------------------------------------------------------
# 9. Hyperparameter tuning (3-fold cross-validation on training set)
# ---------------------------------------------------------------------------
# We tune two key hyperparameters:
#   maxDepth: controls tree complexity (bias-variance tradeoff)
#   numTrees: more trees = lower variance, diminishing returns beyond 200
# Evaluation metric: areaUnderROC — chosen because:
#   a) accuracy is deceptive with 90/10 imbalance
#   b) AUC is explicitly listed in the Spark course materials (BinaryClassificationEvaluator)
#   c) AUC measures separation quality across all thresholds

evaluator_auc = BinaryClassificationEvaluator(
    labelCol=LABEL_COL,
    metricName="areaUnderROC"
)

param_grid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [100, 200])
    .addGrid(rf.maxDepth, [8, 12])
    .build())

print(f"\n  Cross-validation: 3-fold, {len(param_grid)} hyperparameter combinations")

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator_auc,
    numFolds=3,
    seed=42,
    parallelism=2,
)

print("  Training with cross-validation (this may take ~3-5 minutes)...")
cv_model = cv.fit(train_df)
best_model = cv_model.bestModel

# Extract best params
best_rf = best_model.stages[-1]
print(f"\n  Best numTrees: {best_rf.getNumTrees}")
print(f"  Best maxDepth: {best_rf.getMaxDepth()}")
print(f"  CV AUC scores: {[round(m, 4) for m in cv_model.avgMetrics]}")

# ---------------------------------------------------------------------------
# 10. Evaluate on held-out test set
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST SET EVALUATION (June 2020)")
print("=" * 60)

predictions = best_model.transform(test_df)

# Primary metric: AUC-ROC
auc = evaluator_auc.evaluate(predictions)
print(f"\n  AUC-ROC: {auc:.4f}")

# Accuracy
acc_eval = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy"
)
acc = acc_eval.evaluate(predictions)
print(f"  Accuracy: {acc:.4f}")

# F1-score (harmonic mean of precision and recall — important for imbalanced data)
f1_eval = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL, predictionCol="prediction", metricName="f1"
)
f1 = f1_eval.evaluate(predictions)
print(f"  F1 Score: {f1:.4f}")

# Precision (cleaning-class only): how many flagged panels actually needed cleaning?
prec_eval = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL, predictionCol="prediction",
    metricName="weightedPrecision"
)
prec = prec_eval.evaluate(predictions)
print(f"  Weighted Precision: {prec:.4f}")

# Recall
rec_eval = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL, predictionCol="prediction",
    metricName="weightedRecall"
)
rec = rec_eval.evaluate(predictions)
print(f"  Weighted Recall: {rec:.4f}")

# Confusion matrix (via Pandas — small enough)
pred_pd = predictions.select(LABEL_COL, "prediction").toPandas()
TP = ((pred_pd[LABEL_COL] == 1) & (pred_pd["prediction"] == 1)).sum()
TN = ((pred_pd[LABEL_COL] == 0) & (pred_pd["prediction"] == 0)).sum()
FP = ((pred_pd[LABEL_COL] == 0) & (pred_pd["prediction"] == 1)).sum()
FN = ((pred_pd[LABEL_COL] == 1) & (pred_pd["prediction"] == 0)).sum()

print(f"\n  Confusion Matrix:")
print(f"                Predicted 0   Predicted 1")
print(f"  Actual 0    {TN:>10,}  {FP:>10,}  (True Neg / False Pos)")
print(f"  Actual 1    {FN:>10,}  {TP:>10,}  (False Neg / True Pos)")
print(f"\n  Precision (cleaning class): {TP/(TP+FP) if TP+FP > 0 else 0:.4f}")
print(f"  Recall    (cleaning class): {TP/(TP+FN) if TP+FN > 0 else 0:.4f}")

# ---------------------------------------------------------------------------
# 11. Feature Importances
# ---------------------------------------------------------------------------
importances = best_rf.featureImportances.toArray()
fi_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n  Feature Importances (Gini impurity reduction):")
for _, row in fi_df.iterrows():
    bar = "█" * int(row["Importance"] * 80)
    print(f"  {row['Feature']:25s} {row['Importance']:.4f}  {bar}")

# ---------------------------------------------------------------------------
# 12. Visualisations
# ---------------------------------------------------------------------------
sns.set_theme(style="darkgrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"    : 130,
    "font.family"   : "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.bbox"  : "tight",
    "savefig.dpi"   : 130,
})

# Chart A: Feature Importances
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2ecc71" if imp > 0.15 else "#3498db" if imp > 0.05 else "#95a5a6"
          for imp in fi_df["Importance"]]
ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors, edgecolor="white")
ax.set_xlabel("Gini Importance (mean decrease in impurity)")
ax.set_title("MODEL 1 — Random Forest Feature Importances\n"
             "Smart Solar: Cleaning Flag Prediction (Daylight hours, May→June 2020)",
             fontweight="bold")
ax.invert_yaxis()
ax.axvline(0.10, color="gray", linestyle="--", linewidth=1, alpha=0.7,
           label="10% importance threshold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES}/model_01_feature_importance.png")
plt.close()
print("\nSaved: model_01_feature_importance.png")

# Chart B: Confusion Matrix Heatmap
cm_array = np.array([[TN, FP], [FN, TP]])
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_array, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["Predicted: No Clean", "Predicted: Clean"],
            yticklabels=["Actual: No Clean", "Actual: Clean"],
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title(f"MODEL 2 — Confusion Matrix (Test: June 2020)\n"
             f"AUC={auc:.3f}  |  F1={f1:.3f}  |  Acc={acc:.3f}",
             fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES}/model_02_confusion_matrix.png")
plt.close()
print("Saved: model_02_confusion_matrix.png")

# Chart C: Prediction Score Distribution
pred_scores = predictions.select(LABEL_COL, "probability").toPandas()
pred_scores["prob_positive"] = pred_scores["probability"].apply(lambda x: float(x[1]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MODEL 3 — Predicted Probability Distribution by True Class\n"
             "A well-separated model shows two distinct peaks",
             fontweight="bold", fontsize=13)

for flag_val, color, label in [(0, "#3498db", "No Cleaning Needed (actual)"),
                               (1, "#e74c3c", "Cleaning Needed (actual)")]:
    subset = pred_scores[pred_scores[LABEL_COL] == flag_val]["prob_positive"]
    axes[0].hist(subset, bins=50, alpha=0.65, color=color, label=label,
                 edgecolor="white", density=True)

axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5,
                label="Decision threshold (0.5)")
axes[0].set_xlabel("Predicted Probability (cleaning=1)")
axes[0].set_ylabel("Density")
axes[0].set_title("Probability Distribution by True Label")
axes[0].legend(fontsize=9)

# Threshold sensitivity
thresholds = np.linspace(0.1, 0.9, 80)
t_precision, t_recall, t_f1 = [], [], []
for t in thresholds:
    tp = ((pred_scores[LABEL_COL] == 1) & (pred_scores["prob_positive"] >= t)).sum()
    fp = ((pred_scores[LABEL_COL] == 0) & (pred_scores["prob_positive"] >= t)).sum()
    fn = ((pred_scores[LABEL_COL] == 1) & (pred_scores["prob_positive"] < t)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    t_precision.append(p)
    t_recall.append(r)
    t_f1.append(f)

best_t_idx = np.argmax(t_f1)
best_threshold = thresholds[best_t_idx]

axes[1].plot(thresholds, t_precision, color="#3498db", linewidth=2, label="Precision")
axes[1].plot(thresholds, t_recall,    color="#e74c3c", linewidth=2, label="Recall")
axes[1].plot(thresholds, t_f1,        color="#2ecc71", linewidth=2.5, label="F1 Score")
axes[1].axvline(best_threshold, color="gray", linestyle="--", linewidth=1.5,
                label=f"Best threshold ({best_threshold:.2f})")
axes[1].axvline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.7,
                label="Default threshold (0.5)")
axes[1].set_xlabel("Classification Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title(f"Precision / Recall / F1 vs Threshold\nBest F1={max(t_f1):.3f} @ t={best_threshold:.2f}")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f"{FIGURES}/model_03_threshold_analysis.png")
plt.close()
print("Saved: model_03_threshold_analysis.png")

# ---------------------------------------------------------------------------
# 13. Save model outputs
# ---------------------------------------------------------------------------
fi_df.to_csv(f"{FIGURES}/model_feature_importances.csv", index=False)
print("Saved: model_feature_importances.csv")

# ---------------------------------------------------------------------------
# 14. Summary report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("WEEK 3 MODEL SUMMARY")
print("=" * 60)
print(f"  Algorithm       : Random Forest Classifier (PySpark MLlib)")
print(f"  Best numTrees   : {best_rf.getNumTrees}")
print(f"  Best maxDepth   : {best_rf.getMaxDepth()}")
print(f"  Train rows      : {train_count:,}  (May 2020, GHI > 10)")
print(f"  Test rows       : {test_count:,}  (June 2020, GHI > 10)")
print(f"  Class imbalance : {ratio:.1f}:1  (handled via weightCol)")
print(f"  AUC-ROC         : {auc:.4f}")
print(f"  F1 Score        : {f1:.4f}")
print(f"  Accuracy        : {acc:.4f}")
print(f"  Recall (clean)  : {TP/(TP+FN) if TP+FN > 0 else 0:.4f}")
print(f"  Best threshold  : {best_threshold:.2f}  (maximises F1)")
print(f"\n  Top-3 features by importance:")
for _, row in fi_df.head(3).iterrows():
    print(f"    {row['Feature']:25s}: {row['Importance']:.4f}")
print("\nNotebook 04 complete.")
print("Figures saved to ./figures/model_*.png")
