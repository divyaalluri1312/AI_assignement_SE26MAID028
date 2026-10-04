last_average = (
    tracking_df.tail(1000)["agent_sleep_score"].mean()
)

print("\n==============================")
print("SLEEP SCORE TRACKING")
print("==============================")

print(
    "Average of first 1000 observations:",
    round(first_average, 2)
)

print(
    "Average of last 1000 observations:",
    round(last_average, 2)
)

if last_average > first_average + 1:
    print("Trend: Sleep score is improving.")

elif last_average < first_average - 1:
    print("Trend: Sleep score is decreasing.")

else:
    print("Trend: Sleep score is relatively stable.")

# ============================================================
# STEP 21: FINAL INTELLIGENT AGENT
# ============================================================

def intelligent_sleep_agent(row):
    score = row["agent_sleep_score"]
    stress = row["stress_score"]
    mental_health = row["mental_health_condition"]

    print("\n======================================")
    print("INTELLIGENT SLEEP AGENT")
    print("======================================")

    print("Sleep Score:", round(score, 2))
    print(
        "Sleep Duration:",
        row["sleep_duration_hrs"],
        "hours"
    )
    print(
        "Light Sleep:",
        round(row["light_sleep_percentage"], 2),
        "%"
    )
    print(
        "REM Sleep:",
        round(row["rem_percentage"], 2),
        "%"
    )
    print(
        "Deep Sleep:",
        round(row["deep_sleep_percentage"], 2),
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


# Test the final agent using the first record.
intelligent_sleep_agent(df.iloc[0])

# ============================================================
# STEP 22: SAVE RESULTS
# ============================================================

output_file = "intelligent_sleep_agent_results.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nResults saved as:", output_file)
