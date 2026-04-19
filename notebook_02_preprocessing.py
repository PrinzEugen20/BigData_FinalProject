# notebook_02_preprocessing.py
# Smart Solar Management - Big Data Project
#
# Ingests the three raw datasets and produces a single, analytics-ready
# Parquet file at data/processed/solar_preprocessed.parquet.
#
# The Kaggle plant data covers 34 days in May-June 2020 at Gandikota, India.
# NASA POWER timestamps are in UTC. Before joining, they are shifted to
# India Standard Time (UTC+5:30) so that a noon IST reading in the plant data
# aligns with an actual noon-IST irradiance value, not an early-morning one.
# Without this shift the GHI-to-DC_POWER correlation is spuriously negative.
#
# Join strategy:
#   Step 7a - Generation and Weather share exact plant timestamps, so they
#             are joined on DATETIME + PLANT_ID.
#   Step 7b - The Gandikota 2020 NASA subset is aggregated to hourly means
#             by (MONTH, DAY, HOUR). This lets each 15-minute plant reading
#             pick up the correct hourly irradiance without collapsing the
#             intra-day variance that drives DC_POWER.
#
# Dependencies: pip install pyspark pvlib pandas pyarrow

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql.functions import mean as smean
import os
import glob

spark = (SparkSession.builder
         .appName("SmartSolar_02_Preprocessing")
         .config("spark.sql.repl.eagerEval.enabled", True)
         .getOrCreate())

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
gen_p1 = spark.read.csv("data/raw/Plant_1_Generation_Data.csv",
                         header=True, inferSchema=True)
gen_p2 = spark.read.csv("data/raw/Plant_2_Generation_Data.csv",
                         header=True, inferSchema=True)
wx_p1  = spark.read.csv("data/raw/Plant_1_Weather_Sensor_Data.csv",
                         header=True, inferSchema=True)
wx_p2  = spark.read.csv("data/raw/Plant_2_Weather_Sensor_Data.csv",
                         header=True, inferSchema=True)

gen_raw     = gen_p1.unionByName(gen_p2)
weather_raw = wx_p1.unionByName(wx_p2)

# Load all per-city NASA CSVs and tag each row with its city and year so
# the downstream filter (Step 7b) can isolate Gandikota 2020.
nasa_files = [f for f in sorted(glob.glob("data/raw/nasa_power/*.csv"))
              if "combined" not in f]
nasa_parts = []
for fpath in nasa_files:
    city = os.path.basename(fpath).split("_")[0]
    year = int(os.path.basename(fpath).split("_")[1].replace(".csv", ""))
    df   = (spark.read
                 .option("header", True)
                 .option("inferSchema", True)
                 .csv(fpath)
            .withColumn("CITY", F.lit(city))
            .withColumn("YEAR", F.lit(year)))
    nasa_parts.append(df)

nasa_raw = nasa_parts[0]
for p in nasa_parts[1:]:
    nasa_raw = nasa_raw.unionByName(p, allowMissingColumns=True)

print("Raw data loaded")
print(f"  Generation : {gen_raw.count():,} rows")
print(f"  Weather    : {weather_raw.count():,} rows")
print(f"  NASA POWER : {nasa_raw.count():,} rows")

# ---------------------------------------------------------------------------
# Step 1 - Remove duplicates
# ---------------------------------------------------------------------------
gen_df     = gen_raw.dropDuplicates()
weather_df = weather_raw.dropDuplicates()
nasa_df    = nasa_raw.dropDuplicates()

print("\nStep 1 - Duplicates removed")
for name, raw, clean in [("Generation", gen_raw, gen_df),
                          ("Weather",    weather_raw, weather_df),
                          ("NASA POWER", nasa_raw, nasa_df)]:
    removed = raw.count() - clean.count()
    print(f"  {name}: {removed} duplicate rows removed")

# ---------------------------------------------------------------------------
# Step 2 - Drop rows with null timestamps
# ---------------------------------------------------------------------------
# A row with no timestamp cannot participate in any time-based join and has
# no analytical value. Dropping is preferable to imputation here.
gen_df  = gen_df.na.drop(subset=["DATE_TIME"])
nasa_df = nasa_df.filter(F.col("DATETIME").isNotNull())
print("\nStep 2 - Null timestamps dropped")

# ---------------------------------------------------------------------------
# Step 3 - Fill missing numerical values with column mean
# ---------------------------------------------------------------------------
def fill_with_means(df, cols):
    """Replace nulls with the column mean to preserve row count.

    Sensor drop-outs in time-series data are common (network hiccups,
    sensor resets). Deleting affected rows would create temporal gaps
    that distort time-of-day aggregations later. Mean imputation is
    conservative and keeps the distribution shape intact.
    """
    means    = df.select([smean(c).alias(c) for c in cols]).collect()[0]
    fill_map = {c: float(means[c]) for c in cols if means[c] is not None}
    return df.na.fill(fill_map)

gen_df     = fill_with_means(gen_df,
                 ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"])
weather_df = fill_with_means(weather_df,
                 ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"])
nasa_df    = fill_with_means(nasa_df,
                 ["ghi", "dhi", "dni", "temp_air", "wind_speed",
                  "SOLAR_ZENITH_ANGLE", "OPTIMAL_TILT"])

print("Step 3 - Missing values filled with column means")

# ---------------------------------------------------------------------------
# Step 4 - Parse timestamps into a unified DATETIME column
# ---------------------------------------------------------------------------
# Plant 1 uses 'dd-MM-yyyy HH:mm'; Plant 2 uses 'yyyy-MM-dd HH:mm:ss'.
# coalesce tries the first format and falls back to the second, so both
# plants are handled in a single pass without branching.
gen_df = gen_df.withColumn(
    "DATETIME", F.coalesce(
        F.to_timestamp(F.col("DATE_TIME"), "dd-MM-yyyy HH:mm"),
        F.to_timestamp(F.col("DATE_TIME"), "yyyy-MM-dd HH:mm:ss"),
    )
)
weather_df = weather_df.withColumn(
    "DATETIME", F.coalesce(
        F.to_timestamp(F.col("DATE_TIME"), "dd-MM-yyyy HH:mm"),
        F.to_timestamp(F.col("DATE_TIME"), "yyyy-MM-dd HH:mm:ss"),
    )
)

# NASA POWER timestamps come out of pvlib in UTC. Shift them to IST so
# the HOUR extracted in Step 7b matches the IST hour in the Kaggle data.
# Without this shift a UTC noon reading joins to IST 17:30, which makes
# peak irradiance appear to correspond to late-afternoon generation.
nasa_df = nasa_df.withColumn(
    "DATETIME", F.from_utc_timestamp(F.col("DATETIME"), "Asia/Kolkata")
)

# MONTH is used for the seasonal tilt analysis in EDA 5 and for the join key.
for df_name in ["gen_df", "weather_df", "nasa_df"]:
    exec(f'{df_name} = {df_name}.withColumn("MONTH", F.month("DATETIME"))')

print("Step 4 - Timestamps parsed and converted to IST, MONTH column added")

# ---------------------------------------------------------------------------
# Step 5 - Feature engineering
# ---------------------------------------------------------------------------
# EFFICIENCY = AC_POWER / DC_POWER
# Measures how much of the raw panel power survives inverter conversion.
# When dust accumulates, DC_POWER drops while AC_POWER follows, so the
# ratio is a cleaner soiling signal than raw power alone.
# Reference: Fouad et al. (2017) identify this ratio as a standard PV
# performance index.
gen_df = gen_df.withColumn(
    "EFFICIENCY",
    F.when(F.col("DC_POWER") > 0.0,
           F.col("AC_POWER") / F.col("DC_POWER"))
     .otherwise(F.lit(0.0))
     .cast(FloatType())
)

# SOILING_PROXY = 1 - (DAILY_YIELD / plant_average_daily_yield)
# A value of 0 means the panel produced exactly the plant average.
# Positive values indicate underperformance; dust is the primary cause
# during dry periods between rain events.
# Reference: Mani & Pillai (2010) document this systematic yield degradation.
plant_avg = gen_df.groupBy("PLANT_ID").agg(smean("DAILY_YIELD").alias("AVG_YIELD"))
gen_df = gen_df.join(plant_avg, on="PLANT_ID", how="left")
gen_df = gen_df.withColumn(
    "SOILING_PROXY",
    F.when(F.col("AVG_YIELD") > 0,
           1.0 - (F.col("DAILY_YIELD") / F.col("AVG_YIELD")))
     .otherwise(F.lit(0.0))
     .cast(FloatType())
).drop("AVG_YIELD")

# OPTIMAL_TILT was computed during download (= 90 - solar zenith, degrees).
# Clip to [0, 90] to discard physically impossible angles that can appear
# when the sun is below the horizon (zenith > 90).
nasa_df = nasa_df.withColumn(
    "OPTIMAL_TILT",
    F.greatest(F.lit(0.0),
               F.least(F.lit(90.0), F.col("OPTIMAL_TILT"))).cast(FloatType())
)

# CLEANING_FLAG - binary cleaning recommendation.
# Fires when a panel is producing more than 10% below the plant average
# AND its inverter efficiency has dropped below 0.95.
# The 0.95 threshold comes from Panat & Varanasi (2022), who showed that
# waterless electrostatic cleaning recovers up to 95% of lost output,
# making this the practical point at which triggering a cleaning cycle
# is cost-effective.
gen_df = gen_df.withColumn(
    "CLEANING_FLAG",
    F.when(
        (F.col("SOILING_PROXY") > 0.10) & (F.col("EFFICIENCY") < 0.95),
        F.lit(1)
    ).otherwise(F.lit(0)).cast(IntegerType())
)

print("Step 5 - Features engineered: EFFICIENCY, SOILING_PROXY, OPTIMAL_TILT, CLEANING_FLAG")

# ---------------------------------------------------------------------------
# Step 6 - Retain only project-relevant columns
# ---------------------------------------------------------------------------
gen_clean = gen_df.select(
    "DATETIME", "MONTH", "PLANT_ID", "SOURCE_KEY",
    "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD",
    "EFFICIENCY", "SOILING_PROXY", "CLEANING_FLAG"
)

weather_clean = weather_df.select(
    "DATETIME", "PLANT_ID", "SOURCE_KEY",
    "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"
)

nasa_clean = nasa_df.select(
    "DATETIME", "MONTH", "CITY", "YEAR",
    F.col("ghi").alias("GHI"),
    F.col("dhi").alias("DHI"),
    F.col("dni").alias("DNI"),
    F.col("temp_air").alias("AIR_TEMPERATURE"),
    F.col("wind_speed").alias("WIND_SPEED"),
    "SOLAR_ZENITH_ANGLE", "OPTIMAL_TILT", "LATITUDE", "LONGITUDE"
)

print("Step 6 - Columns filtered to project-relevant set")

# ---------------------------------------------------------------------------
# Step 7 - Merge datasets
# ---------------------------------------------------------------------------

# 7a. Join generation records with weather sensor readings.
# SOURCE_KEY is dropped from the weather frame because it identifies the
# sensor unit, whereas in the generation frame it identifies the inverter.
# Keeping both would create ambiguity; PLANT_ID + DATETIME is sufficient.
gen_weather = gen_clean.join(
    weather_clean.drop("SOURCE_KEY"),
    on=["DATETIME", "PLANT_ID"],
    how="left"
)

# 7b. Prepare the NASA join table.
# Only Gandikota 2020 data is used here. The Kaggle plant operates in 2020
# at Gandikota, so this subset provides the closest physical match.
# Readings are aggregated to hourly means by (MONTH, DAY, HOUR). This is
# necessary because the plant logs at 15-minute intervals while NASA POWER
# is hourly; averaging preserves the irradiance shape across the day.
nasa_for_join = (
    nasa_clean
    .filter((F.col("CITY") == "Gandikota") & (F.col("YEAR") == 2020))
    .withColumn("MONTH", F.month("DATETIME"))
    .withColumn("DAY",   F.dayofmonth("DATETIME"))
    .withColumn("HOUR",  F.hour("DATETIME"))
    .groupBy("MONTH", "DAY", "HOUR")
    .agg(
        F.avg("GHI").alias("GHI"),
        F.avg("DHI").alias("DHI"),
        F.avg("DNI").alias("DNI"),
        F.avg("AIR_TEMPERATURE").alias("AIR_TEMPERATURE"),
        F.avg("WIND_SPEED").alias("WIND_SPEED"),
        F.avg("SOLAR_ZENITH_ANGLE").alias("SOLAR_ZENITH_ANGLE"),
        F.avg("OPTIMAL_TILT").alias("OPTIMAL_TILT"),
        F.avg("LATITUDE").alias("LATITUDE"),
        F.avg("LONGITUDE").alias("LONGITUDE"),
    )
)

# Add the same time keys to the generation side so the join can match them.
gen_weather = (gen_weather
               .withColumn("DAY",  F.dayofmonth("DATETIME"))
               .withColumn("HOUR", F.hour("DATETIME")))

# 7c. Left-join on (MONTH, DAY, HOUR). Using left ensures every generation
# row is kept even if NASA data for that specific hour is missing.
final_df = (gen_weather
            .join(nasa_for_join, on=["MONTH", "DAY", "HOUR"], how="left")
            .drop("DAY", "HOUR"))

print(f"\nStep 7 - Datasets merged")
print(f"  Final row count : {final_df.count():,}")
print(f"  Final columns   : {len(final_df.columns)}")
print(f"  Columns         : {final_df.columns}")

# ---------------------------------------------------------------------------
# Step 8 - Save preprocessed dataset
# ---------------------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)

# Parquet is the primary output. It is compressed, schema-aware, and natively
# supported by PySpark, Pandas, and ML frameworks. The CSV backup is kept for
# quick inspection without a Spark session.
final_df.write.mode("overwrite").parquet("data/processed/solar_preprocessed.parquet")
print("\nSaved: data/processed/solar_preprocessed.parquet")

final_df.coalesce(1).write.mode("overwrite") \
    .option("header", True).csv("data/processed/solar_preprocessed_csv")
print("Saved: data/processed/solar_preprocessed_csv/")

# ---------------------------------------------------------------------------
# Step 9 - Validation report
# ---------------------------------------------------------------------------
from pyspark.sql.functions import count as cnt, when, col

final_count = final_df.count()
raw_count   = gen_raw.count() + weather_raw.count() + nasa_raw.count()

print("\n" + "=" * 60)
print("PREPROCESSING VALIDATION REPORT")
print(f"  Total raw rows in: {raw_count:,}")
print(f"  Total rows out   : {final_count:,}")
print(f"  Total columns    : {len(final_df.columns)}")
print(f"  500K req (raw)   : {'PASS' if raw_count >= 500_000 else 'FAIL'} "
      f"({raw_count:,} rows)")

null_final = final_df.select([
    cnt(when(col(c).isNull(), c)).alias(c)
    for c in final_df.columns
])
print("\n  Remaining nulls after preprocessing:")
null_final.show(truncate=False)

flag_dist = final_df.groupBy("CLEANING_FLAG").count().orderBy("CLEANING_FLAG")
print("  CLEANING_FLAG distribution:")
flag_dist.show()

print("Notebook 02 complete. Proceed to notebook_03_eda.py")
