Assignment 10 - Air Quality Index (AQI) Calculation
Overview
This assignment calculates Air Quality Index (AQI) values from the
provided city_day.csv air-quality dataset.

The program calculates pollutant sub-indices for:

PM2.5
PM10
SO2
NOx
NH3
CO
O3
The final calculated AQI is the maximum valid pollutant sub-index,
subject to the minimum-data rule.

Important Dataset Note
The original tutorial notebook supplied for this work is based on
hourly station data. Its methodology uses:

24-hour averages for PM2.5, PM10, SO2, NOx and NH3.
Maximum 8-hour values for CO and O3.
At least one of PM2.5 or PM10.
At least three of the seven pollutant sub-indices.
The supplied city_day.csv contains one daily record per city and does
not contain the underlying hourly observations. Therefore, this
adapted program uses the available daily CO and O3 values when
calculating the city-day estimate. It also retains the dataset's
existing AQI column for comparison.

Consequently, the calculated AQI should not be expected to match the
dataset's stored AQI exactly for every row. The program reports the
match percentage, mean absolute error and correlation so the difference
is transparent.

AQI Categories
AQI	Category
0-50	Good
51-100	Satisfactory
101-200	Moderate
201-300	Poor
301-400	Very Poor
Above 400	Severe
These categories follow the CPCB AQI framework.

Program Workflow
Upload/load city_day.csv.
Validate required columns.
Convert pollutant values to numeric form.
Calculate pollutant-specific sub-indices using linear interpolation.
Apply the minimum-data rule.
Select the maximum valid sub-index as calculated AQI.
Assign the AQI category.
Compare calculated AQI with the AQI already present in the dataset.
Generate AQI distribution and city-level summary visualizations.
Save the processed output as aqi_calculation_results.csv.
Output
The program generates:

aqi_calculation_results.csv
The output contains the original data together with pollutant sub-indices,
the calculated AQI, available-sub-index count, and calculated AQI category.

Running in Google Colab
Open the solution.py code in Google Colab.
Run the program from the beginning.
Upload city_day.csv when prompted.
Wait for the calculations and plots to finish.
The result CSV will be generated and downloaded automatically.
Repository Structure
10_AQI_Calculation/
├── solution.py
└── README.md
The original dataset does not have to be committed to GitHub unless the
professor specifically requires it. The generated result CSV is also
reproducible and does not need to be committed unless required.

Reference
The AQI methodology follows the Indian CPCB framework and the
methodology described in the supplied AQI tutorial.
