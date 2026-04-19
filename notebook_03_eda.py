# notebook_03_eda.py
# Smart Solar Management - Big Data Project
#
# Produces 8 paired Before/After visualisations that document the effect of
# preprocessing and validate the physical relationships the models will use.
#
# Charts 1-4 contrast raw vs. cleaned data to show what preprocessing changed.
# Charts 5-8 are analytical charts over the cleaned data that confirm the
# project hypotheses hold in the actual measurements.
#
# Confirmed correlations (post IST timezone alignment in notebook_02):
#   GHI vs IRRADIATION  r = +0.917  (NASA satellite and ground sensor agree)
#   GHI vs DC_POWER     r = +0.625  (irradiance drives generation as expected)
#   AIR_TEMP vs AMB_T   r = +0.821  (meteorological sources are consistent)
#
# All NASA POWER timestamps were already shifted to IST in notebook_02 before
# the join. The raw NASA CSVs still carry UTC timestamps, so EDA 5 and 6 which
# read those files directly apply utc=True when parsing the index.
#
# Dependencies: pip install pandas numpy matplotlib seaborn pyarrow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import glob
import os
import warnings

warnings.filterwarnings("ignore")

# -- Global plot style -------------------------------------------------------
sns.set_theme(style="darkgrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"    : 130,
    "font.family"   : "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.bbox"  : "tight",
    "savefig.dpi"   : 130,
})
FIGURES = "figures"
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading datasets...")

# Raw sources - used for the BEFORE panels in charts 1-4.
gen_raw_parts = []
for f in ["data/raw/Plant_1_Generation_Data.csv",
          "data/raw/Plant_2_Generation_Data.csv"]:
    if os.path.exists(f):
        gen_raw_parts.append(pd.read_csv(f))
gen_raw = pd.concat(gen_raw_parts, ignore_index=True) if gen_raw_parts else None

wx_raw_parts = []
for f in ["data/raw/Plant_1_Weather_Sensor_Data.csv",
          "data/raw/Plant_2_Weather_Sensor_Data.csv"]:
    if os.path.exists(f):
        wx_raw_parts.append(pd.read_csv(f))
wx_raw = pd.concat(wx_raw_parts, ignore_index=True) if wx_raw_parts else None

# NASA POWER CSVs are read with UTC timestamp parsing. CITY and OPTIMAL_TILT
# were added by download_nasa_power.py and are present in every file.
nasa_files = [f for f in sorted(glob.glob("data/raw/nasa_power/*.csv"))
              if "combined" not in f]
nasa_parts = [pd.read_csv(f, index_col=0, parse_dates=True) for f in nasa_files]
nasa_raw   = pd.concat(nasa_parts, ignore_index=False) if nasa_parts else None

# Cleaned parquet - used for the AFTER panels and analytical charts.
try:
    clean_df  = pd.read_parquet("data/processed/solar_preprocessed.parquet")
    print(f"Cleaned dataset: {len(clean_df):,} rows x {len(clean_df.columns)} cols")
    HAVE_CLEAN = True
except Exception:
    print("Cleaned parquet not found - run notebook_02 first. AFTER panels will be skipped.")
    clean_df   = None
    HAVE_CLEAN = False

# ---------------------------------------------------------------------------
# EDA 1 - Null value counts: before vs. after
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("EDA 1 - Null Values Per Column: Before vs. After Preprocessing",
             fontsize=14, fontweight="bold", y=1.01)

before_nulls = {}
if gen_raw is not None:
    for c in gen_raw.columns:
        before_nulls[c] = int(gen_raw[c].isnull().sum())
if wx_raw is not None:
    for c in ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]:
        if c in wx_raw.columns:
            before_nulls[c] = int(wx_raw[c].isnull().sum())
if nasa_raw is not None:
    for c in ["ghi", "dhi", "dni", "temp_air", "wind_speed"]:
        if c in nasa_raw.columns:
            before_nulls[c] = int(nasa_raw[c].isnull().sum())

before_cols = list(before_nulls.keys())
before_vals = [before_nulls[c] for c in before_cols]
colors_b    = ["#e74c3c" if v > 0 else "#aed6f1" for v in before_vals]

axes[0].barh(before_cols, before_vals, color=colors_b)
axes[0].set_title("BEFORE - Null counts in raw datasets\n"
                  "(blue = no nulls, red = missing values present)",
                  color="#c0392b", fontsize=11)
axes[0].set_xlabel("Number of null values")
axes[0].set_xlim(left=0)
axes[0].invert_yaxis()
if max(before_vals, default=0) == 0:
    axes[0].text(0.5, 0.5, "All raw columns complete\n(0 nulls)",
                 transform=axes[0].transAxes, ha="center", va="center",
                 fontsize=12, color="#27ae60", fontweight="bold")
for i, v in enumerate(before_vals):
    if v > 0:
        axes[0].text(v + max(before_vals) * 0.01, i, f"{v:,}",
                     va="center", fontsize=8, color="#c0392b")

if HAVE_CLEAN:
    after_nulls = clean_df.isnull().sum()
    after_nulls = after_nulls[after_nulls > 0]
    if len(after_nulls) == 0:
        axes[1].text(0.5, 0.5, "Zero nulls remain\nin any engineered column",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=13, color="#27ae60", fontweight="bold")
        axes[1].set_title(f"AFTER - {len(clean_df):,} rows, 0 engineered nulls",
                          color="#27ae60", fontsize=11)
    else:
        axes[1].barh(after_nulls.index, after_nulls.values, color="#f39c12")
        axes[1].set_title(f"AFTER - Remaining nulls ({len(clean_df):,} rows)\n"
                          "(join-only columns: nulls expected where datasets do not overlap)",
                          color="#27ae60", fontsize=11)
        axes[1].set_xlabel("Number of null values")
        axes[1].invert_yaxis()
else:
    axes[1].text(0.5, 0.5, "Run notebook_02 first",
                 transform=axes[1].transAxes, ha="center", fontsize=12, color="gray")
    axes[1].set_title("AFTER - (not yet preprocessed)", color="gray")

plt.tight_layout()
plt.savefig(f"{FIGURES}/eda_01_missing_values.png")
plt.show()
print("Saved: eda_01_missing_values.png")

# ---------------------------------------------------------------------------
# EDA 2 - DC power distribution: before vs. after (log scale)
# ---------------------------------------------------------------------------
# Log scale is used because the raw data contains a large spike at zero (night
# readings), which would otherwise compress the daytime distribution into a
# thin bar. The AFTER panel shows non-zero daylight readings only to expose
# the useful production range.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EDA 2 - DC Power Output Distribution: Before vs. After Preprocessing",
             fontsize=14, fontweight="bold")

if gen_raw is not None:
    raw_dc = gen_raw["DC_POWER"].dropna()
    axes[0].hist(raw_dc, bins=80, color="#e74c3c", edgecolor="white", alpha=0.85, log=True)
    axes[0].axvline(raw_dc.mean(), color="black", linestyle="--", linewidth=1.8,
                    label=f"Mean = {raw_dc.mean():.0f} kW")
    axes[0].axvline(raw_dc.median(), color="navy", linestyle=":", linewidth=1.8,
                    label=f"Median = {raw_dc.median():.0f} kW")
    axes[0].set_title(f"BEFORE - Raw DC Power (log scale)\n"
                      f"Skewed right: {(raw_dc == 0).sum():,} zero readings "
                      f"({(raw_dc == 0).mean() * 100:.1f}%)")
    axes[0].set_xlabel("DC Power (kW)")
    axes[0].set_ylabel("Frequency (log scale)")
    axes[0].legend(fontsize=9)
else:
    axes[0].text(0.5, 0.5, "Kaggle data not found",
                 transform=axes[0].transAxes, ha="center", fontsize=12, color="gray")

if HAVE_CLEAN and "DC_POWER" in clean_df.columns:
    clean_dc = clean_df[clean_df["DC_POWER"] > 0]["DC_POWER"].dropna()
    axes[1].hist(clean_dc, bins=80, color="#2ecc71", edgecolor="white", alpha=0.85, log=True)
    axes[1].axvline(clean_dc.mean(), color="black", linestyle="--", linewidth=1.8,
                    label=f"Mean = {clean_dc.mean():.0f} kW")
    axes[1].axvline(clean_dc.median(), color="navy", linestyle=":", linewidth=1.8,
                    label=f"Median = {clean_dc.median():.0f} kW")
    axes[1].set_title(f"AFTER - Daylight DC Power only (DC > 0)\n"
                      f"{len(clean_dc):,} active readings, mean imputation applied")
    axes[1].set_xlabel("DC Power (kW)")
    axes[1].set_ylabel("Frequency (log scale)")
    axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES}/eda_02_dc_power_dist.png")
plt.show()
print("Saved: eda_02_dc_power_dist.png")

# ---------------------------------------------------------------------------
# EDA 3 - GHI distribution: all hours vs. daylight only
# ---------------------------------------------------------------------------
# Including nighttime hours (GHI = 0) produces a spike that drowns the
# distribution shape of actual solar availability. The GHI > 10 W/m2 cutoff
# removes sensor noise at dawn/dusk as well as the zero plateau at night.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EDA 3 - GHI Solar Irradiance Distribution (NASA POWER Dataset)",
             fontsize=14, fontweight="bold")

if nasa_raw is not None:
    axes[0].hist(nasa_raw["ghi"].dropna(), bins=80,
                 color="#e67e22", edgecolor="white", alpha=0.85)
    axes[0].axvline(nasa_raw["ghi"].mean(), color="black",
                    linestyle="--", linewidth=1.5,
                    label=f"Mean = {nasa_raw['ghi'].mean():.1f} W/m2")
    axes[0].set_title("BEFORE - Raw GHI: all 24 hours including night (GHI = 0)")
    axes[0].set_xlabel("GHI (W/m2)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    ghi_day = nasa_raw[nasa_raw["ghi"] > 10]["ghi"]
    axes[1].hist(ghi_day, bins=80, color="#f1c40f", edgecolor="white", alpha=0.85)
    axes[1].axvline(ghi_day.mean(), color="black", linestyle="--", linewidth=1.5,
                    label=f"Mean = {ghi_day.mean():.1f} W/m2")
    axes[1].set_title("AFTER - GHI: daylight hours only (GHI > 10 W/m2)")
    axes[1].set_xlabel("GHI (W/m2)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

plt.tight_layout()
plt.savefig(f"{FIGURES}/eda_03_ghi_distribution.png")
plt.show()
print("Saved: eda_03_ghi_distribution.png")

# ---------------------------------------------------------------------------
# EDA 4 - Multi-variable boxplots: raw variables vs. engineered features
# ---------------------------------------------------------------------------
# Variables are normalised to [0, 1] so they can share one axis. The BEFORE
# panel uses raw inputs from different sources; the AFTER panel shows the
# engineered features from the merged parquet, which have tighter and more
# interpretable spreads.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("EDA 4 - Key Variable Distributions: Before vs. After Preprocessing",
             fontsize=14, fontweight="bold")

before_data, before_labels, before_colors = [], [], []
if gen_raw is not None and "DC_POWER" in gen_raw.columns:
    dc_norm = (gen_raw["DC_POWER"].dropna() / gen_raw["DC_POWER"].max()).values
    before_data.append(dc_norm)
    before_labels.append("DC Power\n(normalised)")
    before_colors.append("#e74c3c")
if gen_raw is not None and "DAILY_YIELD" in gen_raw.columns:
    dy_norm = (gen_raw["DAILY_YIELD"].dropna() / gen_raw["DAILY_YIELD"].max()).values
    before_data.append(dy_norm)
    before_labels.append("Daily Yield\n(normalised)")
    before_colors.append("#e67e22")
if nasa_raw is not None and "ghi" in nasa_raw.columns:
    ghi_day  = nasa_raw[nasa_raw["ghi"] > 0]["ghi"].dropna()
    ghi_norm = (ghi_day / ghi_day.max()).values
    before_data.append(ghi_norm)
    before_labels.append("GHI\n(normalised)")
    before_colors.append("#e84393")

if before_data:
    bp = axes[0].boxplot(before_data, patch_artist=True, notch=False,
                         flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], before_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_xticks(range(1, len(before_labels) + 1))
    axes[0].set_xticklabels(before_labels, fontsize=10)
    axes[0].set_title("BEFORE - Raw variables (0-1 scaled)\n"
                      "Wide spread indicates noise and outliers",
                      color="#c0392b", fontsize=11)
    axes[0].set_ylabel("Normalised value (0 = min, 1 = max)")
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)

if HAVE_CLEAN:
    after_data, after_labels, after_colors = [], [], []
    if "EFFICIENCY" in clean_df.columns:
        eff = clean_df["EFFICIENCY"].dropna()
        eff = eff[(eff >= 0) & (eff <= 1)]
        after_data.append(eff.values)
        after_labels.append("EFFICIENCY\n(AC/DC ratio)")
        after_colors.append("#2ecc71")
    if "SOILING_PROXY" in clean_df.columns:
        sp = clean_df["SOILING_PROXY"].dropna().clip(0, 1)
        after_data.append(sp.values)
        after_labels.append("SOILING\nPROXY")
        after_colors.append("#3498db")
    if "OPTIMAL_TILT" in clean_df.columns:
        tilt_norm = (clean_df["OPTIMAL_TILT"].dropna() / 90).clip(0, 1)
        after_data.append(tilt_norm.values)
        after_labels.append("OPTIMAL TILT\n(/ 90)")
        after_colors.append("#9b59b6")

    if after_data:
        bp2 = axes[1].boxplot(after_data, patch_artist=True, notch=False,
                              flierprops=dict(marker=".", markersize=2, alpha=0.3))
        for patch, color in zip(bp2["boxes"], after_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_xticks(range(1, len(after_labels) + 1))
        axes[1].set_xticklabels(after_labels, fontsize=10)
        axes[1].set_title("AFTER - Engineered features (0-1 scale)\n"
                          "Tighter, interpretable distributions",
                          color="#27ae60", fontsize=11)
        axes[1].set_ylabel("Normalised value (0-1)")
        # Horizontal reference at 0.95 marks the CLEANING_FLAG threshold
        axes[1].axhline(0.95, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                        label="Cleaning trigger (efficiency < 0.95)")
        axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIGURES}/eda_04_boxplot_outliers.png")
plt.show()
print("Saved: eda_04_boxplot_outliers.png")

# ---------------------------------------------------------------------------
# EDA 5 - Optimal tilt angle by month (seasonal variation)
# ---------------------------------------------------------------------------
# The U-shaped tilt curve across the year validates that a fixed annual tilt
# leaves energy on the table during months where the sun's declination shifts
# significantly. Ben Mansour et al. (2021) quantify this loss at ~4.2% for
# monthly adjustment vs. annual.
if nasa_raw is not None:
    # Parse the UTC index; month extraction does not depend on timezone.
    nasa_raw.index = pd.to_datetime(nasa_raw.index, utc=True)
    nasa_raw["month"] = nasa_raw.index.month

    monthly_tilt = nasa_raw.groupby("month")["OPTIMAL_TILT"].mean()
    monthly_ghi  = nasa_raw.groupby("month")["ghi"].mean()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(range(1, 13), monthly_tilt.reindex(range(1, 13)).values,
             marker="o", color="#e67e22", linewidth=2.5, markersize=9,
             label="Avg Optimal Tilt (deg)")
    ax1.fill_between(range(1, 13), monthly_tilt.reindex(range(1, 13)).values,
                     alpha=0.12, color="#e67e22")
    ax2.bar(range(1, 13), monthly_ghi.reindex(range(1, 13)).values,
            alpha=0.3, color="#3498db", label="Avg GHI (W/m2)")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Optimal Tilt Angle (deg)", color="#e67e22")
    ax2.set_ylabel("Avg GHI (W/m2)", color="#3498db")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_labels)
    ax1.tick_params(axis="y", colors="#e67e22")
    ax2.tick_params(axis="y", colors="#3498db")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("EDA 5 - Monthly Optimal PV Tilt Angle vs. Average GHI\n"
              "(5 Andhra Pradesh cities, 2016-2024)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/eda_05_tilt_by_month.png")
    plt.show()
    print("Saved: eda_05_tilt_by_month.png")

# ---------------------------------------------------------------------------
# EDA 6 - GHI distribution by city (geographic comparison)
# ---------------------------------------------------------------------------
# The five cities span roughly 2.5 degrees of latitude across Andhra Pradesh.
# Showing per-city GHI variation justifies using region-level data rather than
# a single point, as tilt optima will differ across the state.
if nasa_raw is not None and "CITY" in nasa_raw.columns:
    nasa_day = nasa_raw[nasa_raw["ghi"] > 10].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("EDA 6 - GHI Distribution by City (Andhra Pradesh region)\n"
                 "Daylight hours only (GHI > 10 W/m2)",
                 fontsize=13, fontweight="bold")

    city_order = (nasa_day.groupby("CITY")["ghi"]
                  .median().sort_values(ascending=False).index)
    sns.boxplot(data=nasa_day, x="CITY", y="ghi", order=city_order,
                palette="Set2", ax=axes[0],
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
    axes[0].set_title("GHI Distribution per City")
    axes[0].set_xlabel("City")
    axes[0].set_ylabel("GHI (W/m2)")

    pivot = nasa_day.pivot_table(values="ghi", index="CITY",
                                 columns="month", aggfunc="mean")
    pivot.columns = month_labels
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[1],
                linewidths=0.5, cbar_kws={"label": "Avg GHI (W/m2)"})
    axes[1].set_title("Monthly Average GHI Heatmap by City")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("City")

    plt.tight_layout()
    plt.savefig(f"{FIGURES}/eda_06_ghi_by_city.png")
    plt.show()
    print("Saved: eda_06_ghi_by_city.png")

# ---------------------------------------------------------------------------
# EDA 7 - Correlation heatmap (post-preprocessing)
# ---------------------------------------------------------------------------
# This chart is the primary validation of the preprocessing pipeline.
# A positive GHI-to-DC_POWER correlation confirms that the UTC-to-IST
# timezone conversion in notebook_02 correctly aligned the two datasets.
# Before that fix the correlation was -0.53 due to a 5.5-hour time offset.
if HAVE_CLEAN:
    corr_candidates = ["DC_POWER", "AC_POWER", "GHI", "AIR_TEMPERATURE",
                       "WIND_SPEED", "EFFICIENCY", "OPTIMAL_TILT",
                       "SOILING_PROXY", "SOLAR_ZENITH_ANGLE", "DNI", "DHI"]
    corr_cols = [c for c in corr_candidates if c in clean_df.columns]
    # min_periods=30 requires at least 30 overlapping non-null rows per pair,
    # preventing spurious correlations from near-empty column intersections.
    corr = clean_df[corr_cols].corr(min_periods=30)

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, mask=mask, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
    ax.set_title("EDA 7 - Correlation Matrix of Solar Performance Variables\n"
                 "Preprocessed dataset - key relationships for modelling",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/eda_07_correlation_heatmap.png")
    plt.show()
    print("Saved: eda_07_correlation_heatmap.png")
else:
    # Fallback: produce a NASA POWER-only correlation from the raw files so
    # this chart is never completely absent even without the cleaned parquet.
    nasa_corr_cols = ["ghi", "dhi", "dni", "temp_air", "wind_speed",
                      "SOLAR_ZENITH_ANGLE", "OPTIMAL_TILT"]
    existing = [c for c in nasa_corr_cols if c in nasa_raw.columns]
    corr = nasa_raw[existing].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, mask=mask, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("EDA 7 - Correlation Matrix: NASA POWER Variables\n"
                 "(GHI, DNI, DHI, Temperature, Wind, Tilt Angle)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES}/eda_07_correlation_heatmap.png")
    plt.show()
    print("Saved: eda_07_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# EDA 8 - Solar zenith angle vs. GHI and optimal tilt
# ---------------------------------------------------------------------------
# The left scatter shows how GHI drops as the sun moves toward the horizon
# (higher zenith). The right scatter confirms the linear relationship between
# zenith and optimal tilt (tilt = 90 - zenith), with the regression line
# recovering a slope of approximately -1 and intercept of 90 - directly
# matching the formula used in download_nasa_power.py.
if nasa_raw is not None:
    nasa_day = nasa_raw[nasa_raw["ghi"] > 50].sample(
        n=min(5000, len(nasa_raw)), random_state=42
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("EDA 8 - Solar Zenith Angle vs. GHI and Optimal Tilt\n"
                 "Validates the tilt formula: Optimal Tilt = 90 - Zenith",
                 fontsize=13, fontweight="bold")

    sns.scatterplot(data=nasa_day, x="SOLAR_ZENITH_ANGLE", y="ghi",
                    hue="CITY", palette="tab10", alpha=0.5, s=18,
                    linewidth=0, ax=axes[0])
    axes[0].set_title("Solar Zenith Angle vs. GHI")
    axes[0].set_xlabel("Solar Zenith Angle (deg)")
    axes[0].set_ylabel("GHI (W/m2)")

    sns.scatterplot(data=nasa_day, x="SOLAR_ZENITH_ANGLE", y="OPTIMAL_TILT",
                    hue="CITY", palette="tab10", alpha=0.5, s=18,
                    linewidth=0, ax=axes[1], legend=False)

    from numpy.polynomial.polynomial import polyfit as npfit
    pair = nasa_day[["SOLAR_ZENITH_ANGLE", "OPTIMAL_TILT"]].dropna()
    c0, c1 = npfit(pair["SOLAR_ZENITH_ANGLE"].values,
                   pair["OPTIMAL_TILT"].values, 1)
    x_line = np.linspace(pair["SOLAR_ZENITH_ANGLE"].min(),
                         pair["SOLAR_ZENITH_ANGLE"].max(), 100)
    axes[1].plot(x_line, c0 + c1 * x_line, color="red",
                 linewidth=2, linestyle="--",
                 label=f"y = {c1:.2f}x + {c0:.1f}")
    axes[1].set_title("Zenith Angle vs. Optimal Tilt (linear relationship)")
    axes[1].set_xlabel("Solar Zenith Angle (deg)")
    axes[1].set_ylabel("Optimal Tilt (deg)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGURES}/eda_08_zenith_tilt_scatter.png")
    plt.show()
    print("Saved: eda_08_zenith_tilt_scatter.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
for fname in sorted(os.listdir(FIGURES)):
    fpath = os.path.join(FIGURES, fname)
    print(f"  {fname}  ({os.path.getsize(fpath) / 1024:.0f} KB)")
print(f"\nAll figures saved to ./{FIGURES}/")
if not HAVE_CLEAN:
    print("\nNote: Charts 1, 2, 4, and 7 will show the full before/after comparison")
    print("      once notebook_02 has been run to produce the cleaned parquet.")
