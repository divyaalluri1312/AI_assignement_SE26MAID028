"""
Assignment 9 - Intelligent Sleep Monitoring Agent

This program analyzes the Sleep Health & Daily Performance Dataset
and builds a rule-based intelligent sleep monitoring agent.

Main tasks:
1. Load and inspect the dataset.
2. Derive light-sleep percentage.
3. Convert sleep quality to a 0-100 sleep score.
4. Categorize sleep quality.
5. Calculate a sleep-architecture score.
6. Calculate an overall agent sleep score.
7. Generate an agent decision using sleep score, stress, and
   mental-health condition.
8. Produce statistical summaries and visualizations.
9. Track sleep scores using person_id as the observation sequence.
10. Save the complete results to a CSV file.

The program is designed to run in Google Colab.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


print("Please upload the Sleep Health dataset CSV file.")

if IN_COLAB:
    uploaded = files.upload()

    if not uploaded:
        raise ValueError("No dataset file was uploaded.")

    file_name = next(iter(uploaded))
else:
    file_name = input("Enter the CSV file path: ").strip()

df = pd.read_csv(file_name)

print("\nDataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 records:")
display(df.head())


# ============================================================
# STEP 2: CHECK DATASET
# ============================================================

print("\nColumn names:")
print(df.columns.tolist())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nDataset information:")
df.info()


# ============================================================
# STEP 3: SELECT IMPORTANT SLEEP VARIABLES
# ============================================================

sleep_columns = [
    "sleep_duration_hrs",
    "sleep_quality_score",
    "rem_percentage",
    "deep_sleep_percentage",
    "wake_episodes_per_night",
    "stress_score",
    "mental_health_condition",
]

print("\nImportant variables for the Sleep Agent:")
display(df[sleep_columns].head())


# ============================================================
# STEP 4: CALCULATE LIGHT SLEEP
# ============================================================

# Light sleep is the remaining percentage after REM and deep sleep.
df["light_sleep_percentage"] = (
    100
    - df["rem_percentage"]
    - df["deep_sleep_percentage"]
)

# Avoid impossible negative percentages.
df["light_sleep_percentage"] = (
    df["light_sleep_percentage"].clip(lower=0)
)

print("\nSleep architecture:")
display(
    df[
        [
            "sleep_duration_hrs",
            "rem_percentage",
            "deep_sleep_percentage",
            "light_sleep_percentage",
            "wake_episodes_per_night",
        ]
    ].head()
)


# ============================================================
# STEP 5: CONVERT SLEEP QUALITY TO 0-100 SLEEP SCORE
# ============================================================

# sleep_quality_score is on a 1-10 scale.
df["sleep_score"] = (
    df["sleep_quality_score"] * 10
)

print("\nSleep Score:")
display(
    df[
        [
            "sleep_quality_score",
            "sleep_score",
        ]
    ].head()
)


# ============================================================
# STEP 6: SLEEP SCORE CATEGORIES
# ============================================================

def sleep_category(score):
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Moderate"
    else:
        return "Poor"


df["sleep_category"] = df["sleep_score"].apply(
    sleep_category
)

print("\nSleep score categories:")
print(
    df["sleep_category"].value_counts()
)


# ============================================================
# STEP 7: FOUR IMPORTANT SLEEP FACTORS
# ============================================================

print("\nFour important sleep factors:")
print("""
1. Light Sleep
2. REM Sleep
3. Awake / Sleep Disruption
4. Deep Sleep
""")


# ============================================================
# STEP 8: SLEEP ARCHITECTURE SCORE
# ============================================================

# Reference ranges used in the sleep architecture analysis:
# REM sleep: approximately 20-25%
# Deep sleep: approximately 13-23%
# Light sleep: approximately 50-60%
# Fewer wake episodes indicate less sleep disruption.

def range_score(value, low, high):
    if low <= value <= high:
        return 100

    if value < low:
        return max(
            0,
            100 - ((low - value) / low) * 100
        )

    return max(
        0,
        100 - ((value - high) / high) * 100
    )


df["rem_score"] = df["rem_percentage"].apply(
    lambda value: range_score(
        value,
        20,
        25
    )
)

df["deep_score"] = df["deep_sleep_percentage"].apply(
    lambda value: range_score(
        value,
        13,
        23
    )
)

df["light_score"] = df["light_sleep_percentage"].apply(
    lambda value: range_score(
        value,
        50,
        60
    )
)

# Fewer wake episodes are better.
df["awake_score"] = (
    100
    - (df["wake_episodes_per_night"] / 10) * 100
)

df["awake_score"] = (
    df["awake_score"].clip(0, 100)
)

# Combine the four factors.
df["architecture_score"] = (
    df["light_score"]
    + df["rem_score"]
    + df["awake_score"]
    + df["deep_score"]
) / 4

print("\nSleep architecture score:")
display(
    df[
        [
            "light_sleep_percentage",
            "rem_percentage",
            "deep_sleep_percentage",
            "wake_episodes_per_night",
            "architecture_score",
        ]
    ].head()
)


# ============================================================
# STEP 9: INTELLIGENT AGENT SCORE
# ============================================================

# 70% reported sleep quality + 30% sleep architecture.
df["agent_sleep_score"] = (
    0.70 * df["sleep_score"]
    + 0.30 * df["architecture_score"]
)

df["agent_sleep_score"] = (
    df["agent_sleep_score"].clip(0, 100)
)


# ============================================================
# STEP 10: INTELLIGENT AGENT DECISION
# ============================================================

def sleep_agent(
    score,
    stress,
    mental_health
):
    if score < 50:
        return (
            "Poor sleep - improve sleep routine"
        )

    if stress >= 8:
        return (
            "High stress detected - focus on "
            "relaxation and sleep routine"
        )

    if (
        mental_health != "Healthy"
        and score < 65
    ):
        return (
            "Mental health may be affecting sleep - "
            "monitor sleep and stress"
        )

    if score >= 80:
        return (
            "Excellent sleep - maintain current routine"
        )

    if score >= 65:
        return (
            "Good sleep - maintain healthy habits"
        )

    return (
        "Moderate sleep - consider improving "
        "sleep habits"
    )


df["agent_decision"] = df.apply(
    lambda row: sleep_agent(
        row["agent_sleep_score"],
        row["stress_score"],
        row["mental_health_condition"],
    ),
    axis=1,
)


# ============================================================
# STEP 11: DISPLAY AGENT RESULTS
# ============================================================

result_columns = [
    "person_id",
    "sleep_duration_hrs",
    "sleep_quality_score",
    "light_sleep_percentage",
    "rem_percentage",
    "deep_sleep_percentage",
    "wake_episodes_per_night",
    "stress_score",
    "mental_health_condition",
    "agent_sleep_score",
    "agent_decision",
]

print("\nINTELLIGENT SLEEP AGENT RESULTS:")
display(
    df[result_columns].head(15)
)


# ============================================================
# STEP 12: OVERALL SLEEP STATISTICS
# ============================================================

print("\n==============================")
print("OVERALL SLEEP STATISTICS")
print("==============================")

print(
    "Average Sleep Score:",
    round(
        df["sleep_score"].mean(),
        2
    )
)

print(
    "Average Agent Sleep Score:",
    round(
        df["agent_sleep_score"].mean(),
        2
    )
)

print(
    "Average Sleep Duration:",
    round(
        df["sleep_duration_hrs"].mean(),
        2
    ),
    "hours"
)

print(
    "Average REM Sleep:",
    round(
        df["rem_percentage"].mean(),
        2
    ),
    "%"
)

print(
    "Average Deep Sleep:",
    round(
        df["deep_sleep_percentage"].mean(),
        2
    ),
    "%"
)

print(
    "Average Light Sleep:",
    round(
        df["light_sleep_percentage"].mean(),
        2
    ),
    "%"
)

print(
    "Average Wake Episodes:",
    round(
        df["wake_episodes_per_night"].mean(),
        2
    )
)


# ============================================================
# STEP 13: SLEEP SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["sleep_score"],
    bins=20,
    edgecolor="black"
)

plt.title(
    "Distribution of Sleep Score"
)
plt.xlabel("Sleep Score")
plt.ylabel("Number of People")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 14: FOUR SLEEP FACTORS
# ============================================================

factor_means = [
    df["light_sleep_percentage"].mean(),
    df["rem_percentage"].mean(),
    df["wake_episodes_per_night"].mean(),
    df["deep_sleep_percentage"].mean(),
]

factor_names = [
    "Light Sleep %",
    "REM Sleep %",
    "Wake Episodes",
    "Deep Sleep %",
]

plt.figure(figsize=(9, 5))

plt.bar(
    factor_names,
    factor_means
)

plt.title(
    "Average Values of Four Sleep Factors"
)
plt.ylabel("Average Value")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 15: MENTAL HEALTH VS SLEEP SCORE
# ============================================================

mental_health_summary = (
    df.groupby(
        "mental_health_condition"
    )["sleep_score"]
    .mean()
    .sort_values(
        ascending=False
    )
)

print(
    "\nAverage Sleep Score by Mental Health Condition:"
)

display(
    mental_health_summary
)

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="mental_health_condition",
    y="sleep_score"
)

plt.title(
    "Mental Health Condition vs Sleep Score"
)

plt.xlabel(
    "Mental Health Condition"
)

plt.ylabel(
    "Sleep Score"
)

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 16: STRESS VS SLEEP SCORE
# ============================================================

# A sample is used only for plotting so the graph remains readable.
sample_df = df.sample(
    min(5000, len(df)),
    random_state=42
)

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=sample_df,
    x="stress_score",
    y="sleep_score",
    alpha=0.4
)

sns.regplot(
    data=sample_df,
    x="stress_score",
    y="sleep_score",
    scatter=False
)

plt.title(
    "Stress Level vs Sleep Score"
)

plt.xlabel(
    "Stress Score"
)

plt.ylabel(
    "Sleep Score"
)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 17: STRESS-SLEEP CORRELATION
# ============================================================

stress_correlation = (
    df[
        [
            "stress_score",
            "sleep_score"
        ]
    ]
    .corr()
    .iloc[0, 1]
)

print(
    "\nCorrelation between Stress and Sleep Score:",
    round(
        stress_correlation,
        3
    )
)


# ============================================================
# STEP 18: MENTAL HEALTH COMPARISON
# ============================================================

mental_health_table = (
    df.groupby(
        "mental_health_condition"
    )
    .agg(
        Average_Sleep_Score=(
            "sleep_score",
            "mean"
        ),
        Average_Stress=(
            "stress_score",
            "mean"
        ),
        Average_Sleep_Duration=(
            "sleep_duration_hrs",
            "mean"
        ),
        Number_of_Records=(
            "person_id",
            "count"
        ),
    )
    .sort_values(
        "Average_Sleep_Score",
        ascending=False
    )
)

print(
    "\nMental Health Analysis:"
)

display(
    mental_health_table
)


# ============================================================
# STEP 19: SLEEP SCORE TRACKING
# ============================================================

# IMPORTANT:
# tracking_df is created here before it is used below.
# This prevents the NameError that occurred when the tracking
# cell was executed without first creating the dataframe.

tracking_df = df.sort_values(
    "person_id"
).copy()

tracking_df["observation_number"] = range(
    1,
    len(tracking_df) + 1
)

tracking_df["rolling_sleep_score"] = (
    tracking_df["agent_sleep_score"]
    .rolling(
        window=100,
        min_periods=1
    )
    .mean()
)

print(
    "\nSleep Score Tracking Data:"
)

display(
    tracking_df[
        [
            "person_id",
            "agent_sleep_score",
            "observation_number",
            "rolling_sleep_score",
        ]
    ].head()
)

# Plot the first 1000 observations.
plot_df = tracking_df.head(1000)

plt.figure(figsize=(12, 5))

plt.plot(
    plot_df["observation_number"],
    plot_df["agent_sleep_score"],
    alpha=0.3,
    label="Sleep Score"
)

plt.plot(
    plot_df["observation_number"],
    plot_df["rolling_sleep_score"],
    linewidth=2,
    label="Rolling Average"
)

plt.title(
    "Sleep Score Tracking"
)

plt.xlabel(
    "Observation Number"
)

plt.ylabel(
    "Agent Sleep Score"
)

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# STEP 20: SLEEP SCORE TREND
# ============================================================

first_average = (
    tracking_df
    .head(1000)["agent_sleep_score"]
    .mean()
)

last_average = (
    tracking_df
    .tail(1000)["agent_sleep_score"]
    .mean()
)

print("\n==============================")
print("SLEEP SCORE TRACKING")
print("==============================")

print(
    "Average of first 1000 observations:",
    round(
        first_average,
        2
    )
)

print(
    "Average of last 1000 observations:",
    round(
        last_average,
        2
    )
)

if last_average > first_average + 1:

    print(
        "Trend: Sleep score is improving."
    )

elif last_average < first_average - 1:

    print(
        "Trend: Sleep score is decreasing."
    )

else:

    print(
        "Trend: Sleep score is relatively stable."
    )


# ============================================================
# STEP 21: FINAL INTELLIGENT SLEEP AGENT
# ============================================================

def intelligent_sleep_agent(row):

    score = row["agent_sleep_score"]
    stress = row["stress_score"]
    mental_health = row[
        "mental_health_condition"
    ]

    print("\n======================================")
    print("INTELLIGENT SLEEP AGENT")
    print("======================================")

    print(
        "Sleep Score:",
        round(score, 2)
    )

    print(
        "Sleep Duration:",
        row["sleep_duration_hrs"],
        "hours"
    )

    print(
        "Light Sleep:",
        round(
            row["light_sleep_percentage"],
            2
        ),
        "%"
    )

    print(
        "REM Sleep:",
        round(
            row["rem_percentage"],
            2
        ),
        "%"
    )

    print(
        "Deep Sleep:",
        round(
            row["deep_sleep_percentage"],
            2
        ),
        "%"
    )

    print(
        "Wake Episodes:",
        row["wake_episodes_per_night"]
    )

    print(
        "Stress Score:",
        row["stress_score"]
    )

    print(
        "Mental Health:",
        mental_health
    )

    print("\nAgent Decision:")

    if score >= 80:

        print(
            "Excellent sleep. Continue the current "
            "healthy routine."
        )

    elif score >= 65:

        print(
            "Good sleep. Maintain healthy sleep habits."
        )

    elif score >= 50:

        print(
            "Moderate sleep. Try to improve sleep "
            "duration and reduce stress."
        )

    else:

        print(
            "Poor sleep. Sleep habits should be improved."
        )

    if stress >= 8:

        print(
            "Additional Warning: High stress level detected."
        )

    if mental_health != "Healthy":

        print(
            "Additional Observation:",
            "A mental health condition is present and "
            "should be considered while interpreting "
            "sleep quality."
        )


# Run the final agent using the first record.
intelligent_sleep_agent(
    df.iloc[0]
)


# ============================================================
# STEP 22: SAVE RESULTS
# ============================================================

output_file = (
    "intelligent_sleep_agent_results.csv"
)

df.to_csv(
    output_file,
    index=False
)

print(
    "\nResults saved as:",
    output_file
)

# Download automatically when running in Google Colab.
if IN_COLAB:
    files.download(
        output_file
    )

print("\n======================================")
print("ASSIGNMENT 9 COMPLETED SUCCESSFULLY")
print("======================================")
