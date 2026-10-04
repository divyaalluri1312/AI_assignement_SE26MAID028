Assignment 9 - Intelligent Sleep Monitoring Agent
Overview
This assignment implements a rule-based intelligent sleep monitoring
agent using the Sleep Health & Daily Performance Dataset.

The program analyzes sleep duration, sleep quality, REM sleep, deep sleep,
light sleep, wake episodes, stress, and mental-health condition.

Main Objectives
Load and inspect the sleep dataset.
Derive light-sleep percentage.
Convert the sleep quality score from a 1-10 scale to a 0-100 sleep score.
Categorize sleep quality.
Calculate a sleep architecture score.
Calculate an overall intelligent-agent sleep score.
Generate an agent decision.
Analyze the relationship between stress, mental health, and sleep score.
Track sleep scores using person_id as the observation sequence.
Save the complete processed dataset as a CSV result.
Dataset
The program expects the Sleep Health & Daily Performance Dataset.

The dataset is not recreated by the program. Place the CSV file in the
same working environment when running locally, or upload it when running
in Google Colab.

Methodology
1. Light Sleep
Light sleep is derived from the remaining percentage after REM and deep
sleep:

Light Sleep % = 100 - REM % - Deep Sleep %
Negative values are clipped to zero.

2. Sleep Score
The dataset's sleep_quality_score is converted from a 1-10 scale to a
0-100 scale:

Sleep Score = Sleep Quality Score × 10
3. Sleep Categories
Score	Category
80-100	Excellent
65-79	Good
50-64	Moderate
Below 50	Poor
4. Sleep Architecture
Four factors are considered:

Light Sleep
REM Sleep
Awake / Sleep Disruption
Deep Sleep
Reference ranges used by the program:

REM: 20-25%
Deep sleep: 13-23%
Light sleep: 50-60%
Fewer wake episodes are preferred.
The four factor scores are averaged to obtain architecture_score.

5. Intelligent Agent Score
The final agent score combines reported sleep quality and sleep
architecture:

Agent Sleep Score =
0.70 × Sleep Score
+
0.30 × Architecture Score
The result is limited to the 0-100 range.

6. Agent Decision
The rule-based agent considers:

overall agent sleep score,
stress score,
mental-health condition.
Possible decisions include:

Excellent sleep
Good sleep
Moderate sleep
Poor sleep
High-stress warning
Mental-health/sleep observation
Analysis and Visualizations
The program generates:

Sleep score distribution
Average values of the four sleep factors
Mental health condition vs sleep score boxplot
Stress level vs sleep score scatter plot with regression line
Mental-health summary table
Sleep-score tracking with a rolling average
It also calculates the correlation between stress and sleep score.

Output
The program creates:

intelligent_sleep_agent_results.csv
This file contains the original dataset columns plus the derived sleep
and intelligent-agent fields.

How to Run in Google Colab
Open the notebook/code in Google Colab.
Run the cells from top to bottom.
Upload the provided sleep dataset when prompted.
Wait for the analysis and visualizations to finish.
The processed result CSV will be generated at the end.
Repository Structure
09_Intelligent_Sleep_Agent/
├── solution.py
├── README.md
├── sleep_health_dataset.csv
└── Intelligent_Sleep_Agent.ipynb
The generated result CSV does not need to be committed to GitHub because
it can be recreated by running the program.

Important Note
The program is an academic, rule-based sleep analysis exercise. Its
generated decisions are computational outputs from the defined rules and
should not be treated as medical diagnoses.
