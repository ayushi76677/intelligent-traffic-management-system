import cv2
import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "traffic_smoothed_movement.csv"
OUTPUT_FILE = "approaching_vehicle_analysis.csv"

# Minimum number of observations required
MIN_OBSERVATIONS = 10

# How much the bounding box must grow
# before we consider the vehicle approaching
MIN_SIZE_GROWTH = 0.08

# Minimum approach score to trigger warning
WARNING_SCORE = 60

# High-risk score
HIGH_RISK_SCORE = 80


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)

print("Loaded observations:", len(df))


# ============================================================
# CALCULATE BOUNDING BOX SIZE
# ============================================================

df["box_width"] = df["x2"] - df["x1"]
df["box_height"] = df["y2"] - df["y1"]

df["box_area"] = (
    df["box_width"] *
    df["box_height"]
)


# ============================================================
# PREPARE COLUMNS
# ============================================================

df["size_growth"] = 0.0
df["size_growth_rate"] = 0.0
df["position_change"] = 0.0
df["approach_score"] = 0.0
df["risk_level"] = "SAFE"
df["position_zone"] = "UNKNOWN"
df["approaching"] = False


# ============================================================
# PROCESS EACH VEHICLE
# ============================================================

results = []

for track_id, vehicle in df.groupby("track_id"):

    vehicle = vehicle.copy()

    if len(vehicle) < MIN_OBSERVATIONS:
        continue

    vehicle = vehicle.sort_values("frame")

    # --------------------------------------------------------
    # BOX SIZE CHANGE
    # --------------------------------------------------------

    first_area = vehicle["box_area"].iloc[0]
    last_area = vehicle["box_area"].iloc[-1]

    if first_area > 0:

        total_growth = (
            (last_area - first_area)
            / first_area
        )

    else:

        total_growth = 0


    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    first_x = vehicle["center_x"].iloc[0]
    last_x = vehicle["center_x"].iloc[-1]

    first_y = vehicle["center_y"].iloc[0]
    last_y = vehicle["center_y"].iloc[-1]

    dx = last_x - first_x
    dy = last_y - first_y


    # --------------------------------------------------------
    # APPROACHING CAMERA
    # --------------------------------------------------------
    #
    # In this particular camera view, vehicles that become
    # significantly larger are likely getting closer.
    #
    # We combine this with downward image movement.
    # --------------------------------------------------------

    if total_growth > MIN_SIZE_GROWTH:

        size_score = min(
            50,
            total_growth * 100
        )

    else:

        size_score = 0


    # --------------------------------------------------------
    # MOVEMENT SCORE
    # --------------------------------------------------------

    movement_score = 0

    if dy > 5:
        movement_score += 20

    if abs(dx) > 5:
        movement_score += 10


    # --------------------------------------------------------
    # SPEED/MOTION SCORE
    # --------------------------------------------------------

    speeds = vehicle[
        "relative_pixel_speed"
    ].dropna()

    if len(speeds) > 0:

        average_speed = speeds.mean()
        maximum_speed = speeds.max()

    else:

        average_speed = 0
        maximum_speed = 0


    speed_score = min(
        20,
        average_speed / 10
    )


    # --------------------------------------------------------
    # FINAL APPROACH SCORE
    # --------------------------------------------------------

    approach_score = (
        size_score +
        movement_score +
        speed_score
    )

    approach_score = min(
        100,
        approach_score
    )


    # --------------------------------------------------------
    # DETERMINE DRIVER-SCREEN POSITION
    # --------------------------------------------------------

    current_x = vehicle["center_x"].iloc[-1]

    # Video width = 1920
    if current_x < 640:

        position_zone = "LEFT"

    elif current_x < 1280:

        position_zone = "CENTER"

    else:

        position_zone = "RIGHT"


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if approach_score >= HIGH_RISK_SCORE:

        risk_level = "HIGH"
        approaching = True

    elif approach_score >= WARNING_SCORE:

        risk_level = "WARNING"
        approaching = True

    else:

        risk_level = "SAFE"
        approaching = False


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    result = {
        "track_id": track_id,
        "vehicle_type": vehicle["vehicle_type"].mode().iloc[0],

        "observations": len(vehicle),

        "first_time": vehicle["time"].iloc[0],
        "last_time": vehicle["time"].iloc[-1],

        "initial_box_area": first_area,
        "final_box_area": last_area,

        "size_growth": total_growth,

        "dx": dx,
        "dy": dy,

        "average_pixel_speed": average_speed,
        "maximum_pixel_speed": maximum_speed,

        "approach_score": round(
            approach_score,
            2
        ),

        "position_zone": position_zone,

        "risk_level": risk_level,

        "approaching": approaching
    }

    results.append(result)


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

result_df = pd.DataFrame(results)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==========================================")
print("APPROACHING VEHICLE ANALYSIS")
print("==========================================")

print(
    "Vehicles analysed:",
    len(result_df)
)

print(
    "Vehicles approaching:",
    result_df["approaching"].sum()
)

print("\nResults:\n")

if len(result_df) > 0:

    print(
        result_df[
            [
                "track_id",
                "vehicle_type",
                "size_growth",
                "average_pixel_speed",
                "approach_score",
                "position_zone",
                "risk_level"
            ]
        ].to_string(index=False)
    )


# ============================================================
# WARNINGS
# ============================================================

print("\n==========================================")
print("WARNINGS")
print("==========================================")

warnings = result_df[
    result_df["approaching"] == True
]

if len(warnings) == 0:

    print("No rapidly approaching vehicles detected.")

else:

    for _, row in warnings.iterrows():

        print(
            f"⚠️ Track {row['track_id']} "
            f"({row['vehicle_type']}) | "
            f"{row['position_zone']} | "
            f"{row['risk_level']} | "
            f"Score: {row['approach_score']}"
        )


print("\nSaved:")
print(OUTPUT_FILE)