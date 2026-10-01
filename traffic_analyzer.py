import pandas as pd
import json

# LOAD TRAFFIC DATA

file_name = "traffic_timeline.csv"

df = pd.read_csv(file_name)

print("\n===== TRAFFIC DATA =====")
print(df)

# BASIC CALCULATIONS

total_vehicles = df["Total Vehicles"].sum()

average_vehicles = df["Total Vehicles"].mean()

peak_index = df["Total Vehicles"].idxmax()
peak_interval = df.loc[peak_index, "Time Interval"]
peak_vehicles = df.loc[peak_index, "Total Vehicles"]


# VEHICLE COMPOSITION

total_cars = df["Cars"].sum()
total_motorcycles = df["Motorcycles"].sum()
total_buses = df["Buses"].sum()
total_trucks = df["Trucks"].sum()

total_detected = (
    total_cars +
    total_motorcycles +
    total_buses +
    total_trucks
)

motorcycle_percentage = (total_motorcycles / total_detected) * 100
heavy_vehicle_percentage = (
    (total_buses + total_trucks) / total_detected
) * 100

# HIGH TRAFFIC ANALYSIS

high_traffic_intervals = df[
    df["Traffic Level"] == "HIGH"
]

number_of_high_intervals = len(high_traffic_intervals)


# TRAFFIC PATTERN DETECTION

patterns = []

if number_of_high_intervals >= 2:
    patterns.append(
        "Repeated high-traffic conditions detected."
    )

if motorcycle_percentage >= 40:
    patterns.append(
        "Two-wheelers are a dominant vehicle category."
    )

if heavy_vehicle_percentage >= 20:
    patterns.append(
        "Significant heavy-vehicle presence detected."
    )

if peak_vehicles > average_vehicles:
    patterns.append(
        "A clear traffic peak was detected."
    )

# DISPLAY RESULTS

print("\n===== TRAFFIC ANALYSIS =====")

print("Total vehicles detected:", total_vehicles)

print("Average vehicles per interval:",
      round(average_vehicles, 2))

print("Peak traffic interval:", peak_interval)

print("Vehicles during peak:", peak_vehicles)

print("Motorcycle percentage:",
      round(motorcycle_percentage, 2), "%")

print("Heavy vehicle percentage:",
      round(heavy_vehicle_percentage, 2), "%")

print("\nDetected Patterns:")

for pattern in patterns:
    print("-", pattern)

# SAVE ANALYSIS

analysis = {
    "total_vehicles": int(total_vehicles),
    "average_vehicles_per_interval": round(
        float(average_vehicles), 2
    ),
    "peak_interval": peak_interval,
    "peak_vehicles": int(peak_vehicles),
    "motorcycle_percentage": round(
        float(motorcycle_percentage), 2
    ),
    "heavy_vehicle_percentage": round(
        float(heavy_vehicle_percentage), 2
    ),
    "high_traffic_intervals": int(
        number_of_high_intervals
    ),
    "patterns": patterns
}

with open("analysis_report.json", "w") as file:
    json.dump(analysis, file, indent=4)

print("\nAnalysis saved to analysis_report.json")