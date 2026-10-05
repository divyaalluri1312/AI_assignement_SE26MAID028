"""
Assignment 10 - Air Quality Index (AQI) Calculation

Dataset:
    city_day.csv from the Air Quality Data in India dataset.

This version is adapted to the city-day dataset supplied for the
assignment. Unlike the original hourly tutorial, city_day already
contains one daily record per city, so no hourly rolling windows or
station-level files are required.

Important:
The official CPCB methodology uses 24-hour averages for PM2.5, PM10,
SO2, NOx/NO2 and NH3, and short-term (8-hour) values for CO and O3.
The city_day file does not contain the underlying hourly observations.
Therefore, CO and O3 values in this program are treated as the available
daily values for a reproducible city-day AQI estimate. The AQI column
already present in the dataset is retained as a reference value.

The program:
1. Loads and validates city_day.csv.
2. Calculates pollutant sub-indices.
3. Calculates an AQI as the maximum available sub-index.
4. Applies the minimum-data rule.
5. Assigns an AQI category.
6. Compares the calculated estimate with the dataset AQI.
7. Produces useful summary statistics and plots.
8. Saves the processed results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


print("AIR QUALITY INDEX (AQI) CALCULATION")
print("=" * 55)

if IN_COLAB:
    print("\nUpload city_day.csv")
    uploaded = files.upload()

    if not uploaded:
        raise ValueError("No dataset was uploaded.")

    file_name = next(iter(uploaded))
else:
    default_file = "city_day.csv"

    if os.path.exists(default_file):
        file_name = default_file
    else:
        file_name = input(
            "Enter the path to city_day.csv: "
        ).strip()

df = pd.read_csv(file_name)

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 records:")
display(df.head())


# ============================================================
# STEP 2: DATA VALIDATION
# ============================================================

required_columns = [
    "City",
    "Date",
    "PM2.5",
    "PM10",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing: "
        + ", ".join(missing_columns)
    )

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

numeric_columns = [
    "PM2.5",
    "PM10",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("\nMissing values:")
display(
    df[
        required_columns
    ].isnull().sum()
)


# ============================================================
# STEP 3: AQI SUB-INDEX FUNCTIONS
# ============================================================

def interpolate(
    concentration,
    c_low,
    c_high,
    i_low,
    i_high
):
    """Linear interpolation between AQI breakpoints."""

    return (
        (i_high - i_low)
        / (c_high - c_low)
    ) * (
        concentration - c_low
    ) + i_low


def pm25_subindex(x):
    if pd.isna(x):
        return np.nan
    if x <= 30:
        return interpolate(x, 0, 30, 0, 50)
    if x <= 60:
        return interpolate(x, 30, 60, 51, 100)
    if x <= 90:
        return interpolate(x, 60, 90, 101, 200)
    if x <= 120:
        return interpolate(x, 90, 120, 201, 300)
    if x <= 250:
        return interpolate(x, 120, 250, 301, 400)
    return interpolate(x, 250, 500, 401, 500)


def pm10_subindex(x):
    if pd.isna(x):
        return np.nan
    if x <= 50:
        return interpolate(x, 0, 50, 0, 50)
    if x <= 100:
        return interpolate(x, 50, 100, 51, 100)
    if x <= 250:
        return interpolate(x, 100, 250, 101, 200)
    if x <= 350:
        return interpolate(x, 250, 350, 201, 300)
    if x <= 430:
        return interpolate(x, 350, 430, 301, 400)
    return interpolate(x, 430, 500, 401, 500)


def so2_subindex(x):
    if pd.isna(x):
        return np.nan
    if x <= 40:
        return interpolate(x, 0, 40, 0, 50)
    if x <= 80:
        return interpolate(x, 40, 80, 51, 100)
    if x <= 380:
        return interpolate(x, 80, 380, 101, 200)
    if x <= 800:
        return interpolate(x, 380, 800, 201, 300)
    if x <= 1600:
        return interpolate(x, 800, 1600, 301, 400)
    return interpolate(x, 1600, 2000, 401, 500)


def nox_subindex(x):
    if pd.isna(x):
        return np.nan
    if x <= 40:
        return interpolate(x, 0, 40, 0, 50)
    if x <= 80:
        return interpolate(x, 40, 80, 51, 100)
    if x <= 180:
        return interpolate(x, 80, 180, 101, 200)
    if x <= 280:
        return interpolate(x, 180, 280, 201, 300)
    if x <= 400:
        return interpolate(x, 280, 400, 301, 400)
    return interpolate(x, 400, 500, 401, 500)


def nh3_subindex(x):
    if pd.isna(x):
        return np.nan
    if x <= 200:
        return interpolate(x, 0, 200, 0, 50)
    if x <= 400:
        return interpolate(x, 200, 400, 51, 100)
    if x <= 800:
        return interpolate(x, 400, 800, 101, 200)
    if x <= 1200:
        return interpolate(x, 800, 1200, 201, 300)
    if x <= 1800:
        return interpolate(x, 1200, 1800, 301, 400)
    return interpolate(x, 1800, 2000, 401, 500)


def co_subindex(x):
    if pd.isna(x):
        return np.nan
    if x <= 1:
        return interpolate(x, 0, 1, 0, 50)
    if x <= 2:
        return interpolate(x, 1, 2, 51, 100)
    if x <= 10:
        return interpolate(x, 2, 10, 101, 200)
    if x <= 17:
        return interpolate(x, 10, 17, 201, 300)
    if x <= 34:
        return interpolate(x, 17, 34, 301, 400)
    return interpolate(x, 34, 40, 401, 500)


def o3_subindex(x):
    if pd.isna(x):
        return np.nan
    if x <= 50:
        return interpolate(x, 0, 50, 0, 50)
    if x <= 100:
        return interpolate(x, 50, 100, 51, 100)
    if x <= 168:
        return interpolate(x, 100, 168, 101, 200)
    if x <= 208:
        return interpolate(x, 168, 208, 201, 300)
    if x <= 748:
        return interpolate(x, 208, 748, 301, 400)
    return interpolate(x, 748, 800, 401, 500)


# ============================================================
# STEP 4: CALCULATE POLLUTANT SUB-INDICES
# ============================================================

subindex_functions = {
    "PM2.5": pm25_subindex,
    "PM10": pm10_subindex,
    "SO2": so2_subindex,
    "NOx": nox_subindex,
    "NH3": nh3_subindex,
    "CO": co_subindex,
    "O3": o3_subindex,
}

for pollutant, function in subindex_functions.items():
    df[pollutant + "_SubIndex"] = (
        df[pollutant].apply(function)
    )

subindex_columns = [
    pollutant + "_SubIndex"
    for pollutant in subindex_functions
]

print("\nCalculated pollutant sub-indices:")
display(
    df[
        ["City", "Date"]
        + subindex_columns
    ].head()
)


# ============================================================
# STEP 5: CALCULATE AQI
# ============================================================

# The final AQI is the maximum available pollutant sub-index.
df["Available_SubIndices"] = (
    df[subindex_columns]
    .notna()
    .sum(axis=1)
)

df["AQI_calculated"] = (
    df[subindex_columns]
    .max(axis=1)
)

# Minimum-data rule:
# At least one of PM2.5 or PM10 must be available,
# and at least three of the seven pollutants must be available.
pm_available = (
    df["PM2.5_SubIndex"].notna()
    | df["PM10_SubIndex"].notna()
)

df.loc[
    (~pm_available)
    | (df["Available_SubIndices"] < 3),
    "AQI_calculated"
] = np.nan

df["AQI_calculated"] = (
    df["AQI_calculated"].round()
)


# ============================================================
# STEP 6: AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


df["AQI_Category_Calculated"] = (
    df["AQI_calculated"].apply(
        get_aqi_category
    )
)

print("\nAQI category distribution:")
print(
    df["AQI_Category_Calculated"]
    .value_counts(dropna=False)
)


# ============================================================
# STEP 7: DISPLAY CALCULATED AQI
# ============================================================

display_columns = [
    "City",
    "Date",
    "PM2.5",
    "PM10",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "AQI_calculated",
    "AQI_Category_Calculated",
]

print("\nCalculated AQI:")
display(
    df[display_columns].head(20)
)


# ============================================================
# STEP 8: COMPARE WITH DATASET AQI
# ============================================================

if "AQI" in df.columns:

    comparison = df[
        ["AQI", "AQI_calculated"]
    ].dropna()

    if len(comparison) > 0:

        comparison["Difference"] = (
            comparison["AQI_calculated"]
            - comparison["AQI"]
        )

        exact_match = (
            comparison["AQI_calculated"]
            == comparison["AQI"].round()
        ).sum()

        mean_absolute_error = (
            comparison["Difference"]
            .abs()
            .mean()
        )

        correlation = (
            comparison["AQI"]
            .corr(
                comparison["AQI_calculated"]
            )
        )

        print("\n==============================")
        print("AQI VERIFICATION")
        print("==============================")

        print(
            "Comparable rows:",
            len(comparison)
        )

        print(
            "Exact rounded matches:",
            exact_match
        )

        print(
            "Match percentage:",
            round(
                exact_match
                * 100
                / len(comparison),
                2
            ),
            "%"
        )

        print(
            "Mean Absolute Error:",
            round(
                mean_absolute_error,
                2
            )
        )

        print(
            "Correlation:",
            round(
                correlation,
                3
            )
        )

        print(
            "\nNote: The dataset AQI was generated from "
            "underlying hourly/station data. The supplied "
            "city_day file does not contain those hourly "
            "8-hour CO/O3 values, so an exact row-for-row "
            "reproduction is not expected from city_day alone."
        )


# ============================================================
# STEP 9: AQI CATEGORY SUMMARY
# ============================================================

category_summary = (
    df["AQI_Category_Calculated"]
    .value_counts()
    .rename_axis("AQI_Category")
    .reset_index(name="Number_of_Records")
)

print("\nAQI Category Summary:")
display(category_summary)


# ============================================================
# STEP 10: CITY-WISE AQI SUMMARY
# ============================================================

city_summary = (
    df.groupby("City")
    .agg(
        Average_AQI=(
            "AQI_calculated",
            "mean"
        ),
        Maximum_AQI=(
            "AQI_calculated",
            "max"
        ),
        Minimum_AQI=(
            "AQI_calculated",
            "min"
        ),
        Valid_Records=(
            "AQI_calculated",
            "count"
        ),
    )
    .sort_values(
        "Average_AQI",
        ascending=False
    )
)

print("\nCity-wise AQI Summary:")
display(city_summary.head(20))


# ============================================================
# STEP 11: AQI DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["AQI_calculated"].dropna(),
    bins=30,
    edgecolor="black"
)

plt.title(
    "Distribution of Calculated AQI"
)

plt.xlabel("AQI")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: TOP 10 CITIES BY AVERAGE AQI
# ============================================================

top_cities = (
    city_summary
    .head(10)
    .sort_values(
        "Average_AQI"
    )
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_cities.index,
    top_cities["Average_AQI"]
)

plt.title(
    "Top 10 Cities by Average AQI"
)

plt.xlabel(
    "Average Calculated AQI"
)

plt.ylabel(
    "City"
)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 13: SAVE RESULTS
# ============================================================

output_file = (
    "aqi_calculation_results.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(
    "\nResults saved as:",
    output_file
)


# ============================================================
# STEP 14: DOWNLOAD IN GOOGLE COLAB
# ============================================================

if IN_COLAB:
    files.download(
        output_file
    )


print("\n======================================")
print("AQI ASSIGNMENT COMPLETED")
print("======================================")
