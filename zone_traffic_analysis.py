import pandas as pd
import numpy as np
import cv2

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_ground_motion.csv"
ZONE_FILE = "lane_zone_points.npy"

OUTPUT_FILE = "zone_traffic_analysis.csv"

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Loaded observations:", len(df))


# ============================================================
# LOAD ZONE
# ============================================================

zone_points = np.load(
    ZONE_FILE
).astype(np.int32)

print("\nZone points:")

for i, point in enumerate(zone_points):
    print(
        f"P{i + 1}: "
        f"({point[0]}, {point[1]})"
    )


# ============================================================
# CHECK WHETHER VEHICLE IS INSIDE ZONE
# ============================================================

def inside_zone(x, y):

    point = (float(x), float(y))

    result = cv2.pointPolygonTest(
        zone_points,
        point,
        False
    )

    return result >= 0


df["inside_zone"] = [
    inside_zone(x, y)
    for x, y in zip(
        df["bottom_center_x"],
        df["bottom_center_y"]
    )
]


# ============================================================
# ONLY KEEP OBSERVATIONS INSIDE ZONE
# ============================================================

zone_df = df[
    df["inside_zone"] == True
].copy()

print(
    "\nObservations inside zone:",
    len(zone_df)
)


# ============================================================
# VEHICLE COUNT
# ============================================================

vehicle_counts = (
    zone_df
    .groupby(
        "vehicle_type"
    )["track_id"]
    .nunique()
)


cars = vehicle_counts.get(
    "car",
    0
)

motorcycles = vehicle_counts.get(
    "motorcycle",
    0
)

buses = vehicle_counts.get(
    "bus",
    0
)

trucks = vehicle_counts.get(
    "truck",
    0
)

total_vehicles = (
    cars +
    motorcycles +
    buses +
    trucks
)


# ============================================================
# AVERAGE SPEED
# ============================================================

valid_speed = zone_df[
    zone_df["clean_speed"] > 0
]["clean_speed"]

if len(valid_speed) > 0:

    average_speed = valid_speed.mean()
    maximum_speed = valid_speed.max()

else:

    average_speed = 0
    maximum_speed = 0


# ============================================================
# VEHICLE TYPE PERCENTAGES
# ============================================================

if total_vehicles > 0:

    motorcycle_percentage = (
        motorcycles /
        total_vehicles
    ) * 100

    car_percentage = (
        cars /
        total_vehicles
    ) * 100

    bus_percentage = (
        buses /
        total_vehicles
    ) * 100

    truck_percentage = (
        trucks /
        total_vehicles
    ) * 100

else:

    motorcycle_percentage = 0
    car_percentage = 0
    bus_percentage = 0
    truck_percentage = 0


# ============================================================
# TRAFFIC LEVEL
# ============================================================

# Initial prototype thresholds.
# These are not official traffic standards.

if total_vehicles < 10:

    traffic_level = "LOW"

elif total_vehicles < 25:

    traffic_level = "MEDIUM"

else:

    traffic_level = "HIGH"


# ============================================================
# FLOW PER TIME
# ============================================================

zone_df["time_bin"] = (
    zone_df["time"]
    .astype(float)
    .apply(
        lambda x:
        int(x // 10)
    )
)

flow = (
    zone_df
    .groupby("time_bin")["track_id"]
    .nunique()
)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "zone": "ZONE_1",

        "total_vehicles": total_vehicles,

        "cars": cars,

        "motorcycles": motorcycles,

        "buses": buses,

        "trucks": trucks,

        "car_percentage":
            round(
                car_percentage,
                2
            ),

        "motorcycle_percentage":
            round(
                motorcycle_percentage,
                2
            ),

        "bus_percentage":
            round(
                bus_percentage,
                2
            ),

        "truck_percentage":
            round(
                truck_percentage,
                2
            ),

        "average_clean_speed":
            round(
                average_speed,
                3
            ),

        "maximum_clean_speed":
            round(
                maximum_speed,
                3
            ),

        "traffic_level":
            traffic_level
    }
])


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE TIME FLOW
# ============================================================

flow_df = (
    flow
    .reset_index()
)

flow_df.columns = [
    "10_second_interval",
    "unique_vehicles"
]

flow_df.to_csv(
    "zone_traffic_flow.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n==========================================")
print("ZONE TRAFFIC ANALYSIS")
print("==========================================")

print(
    summary.to_string(
        index=False
    )
)

print("\n10-second vehicle flow:")

print(
    flow_df.to_string(
        index=False
    )
)

print("\nCreated:")

print(OUTPUT_FILE)
print("zone_traffic_flow.csv")

print("\nDone.")