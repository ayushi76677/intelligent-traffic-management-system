import pandas as pd
import numpy as np

# ==========================================
# SETTINGS
# ==========================================

INPUT_FILE = "traffic_tracks.csv"
OUTPUT_FILE = "traffic_smoothed_movement.csv"
SUMMARY_FILE = "traffic_smoothed_summary.csv"

# Number of frames used to calculate movement
WINDOW_FRAMES = 10

# Ignore extremely small movements
MOVEMENT_THRESHOLD = 3.0


# ==========================================
# 1. LOAD TRACKING DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

class_names = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

df["vehicle_type"] = df["class"].map(class_names)

df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)


# ==========================================
# 2. REMOVE VERY SHORT TRACKS
# ==========================================

track_counts = df.groupby("track_id").size()

valid_tracks = track_counts[
    track_counts >= WINDOW_FRAMES
].index

df = df[df["track_id"].isin(valid_tracks)].copy()


# ==========================================
# 3. CALCULATE SMOOTHED POSITION
# ==========================================

df["smooth_x"] = (
    df.groupby("track_id")["center_x"]
    .transform(
        lambda x: x.rolling(
            WINDOW_FRAMES,
            min_periods=1
        ).mean()
    )
)

df["smooth_y"] = (
    df.groupby("track_id")["center_y"]
    .transform(
        lambda x: x.rolling(
            WINDOW_FRAMES,
            min_periods=1
        ).mean()
    )
)


# ==========================================
# 4. MOVEMENT OVER WINDOW
# ==========================================

df["previous_smooth_x"] = (
    df.groupby("track_id")["smooth_x"]
    .shift(WINDOW_FRAMES)
)

df["previous_smooth_y"] = (
    df.groupby("track_id")["smooth_y"]
    .shift(WINDOW_FRAMES)
)

df["previous_time"] = (
    df.groupby("track_id")["time"]
    .shift(WINDOW_FRAMES)
)


df["dx"] = (
    df["smooth_x"] -
    df["previous_smooth_x"]
)

df["dy"] = (
    df["smooth_y"] -
    df["previous_smooth_y"]
)


df["time_elapsed"] = (
    df["time"] -
    df["previous_time"]
)


# ==========================================
# 5. CALCULATE RELATIVE SPEED
# ==========================================

df["distance_pixels"] = np.sqrt(
    df["dx"] ** 2 +
    df["dy"] ** 2
)

df["relative_pixel_speed"] = np.where(
    df["time_elapsed"] > 0,
    df["distance_pixels"] /
    df["time_elapsed"],
    0
)


# ==========================================
# 6. DETERMINE DIRECTION
# ==========================================

def calculate_direction(dx, dy):

    if pd.isna(dx) or pd.isna(dy):
        return "UNKNOWN"

    if abs(dx) < MOVEMENT_THRESHOLD and \
       abs(dy) < MOVEMENT_THRESHOLD:
        return "STATIONARY"

    if abs(dx) > abs(dy):

        if dx > 0:
            return "RIGHT"
        else:
            return "LEFT"

    else:

        if dy > 0:
            return "DOWN"
        else:
            return "UP"


df["direction"] = df.apply(
    lambda row: calculate_direction(
        row["dx"],
        row["dy"]
    ),
    axis=1
)


# ==========================================
# 7. CLASS CONSISTENCY
# ==========================================

# Find the most frequently detected class
# for each tracking ID.

dominant_class = (
    df.groupby("track_id")["vehicle_type"]
    .agg(
        lambda x:
        x.mode().iloc[0]
        if not x.mode().empty
        else x.iloc[0]
    )
)

df["vehicle_type"] = df["track_id"].map(
    dominant_class
)


# ==========================================
# 8. SAVE DETAILED DATA
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 9. CREATE VEHICLE SUMMARY
# ==========================================

summary = df.groupby(
    ["track_id", "vehicle_type"]
).agg(

    observations=("track_id", "count"),

    average_relative_speed=(
        "relative_pixel_speed",
        "mean"
    ),

    maximum_relative_speed=(
        "relative_pixel_speed",
        "max"
    ),

    total_distance_pixels=(
        "distance_pixels",
        "sum"
    )

).reset_index()


# ==========================================
# 10. DOMINANT DIRECTION
# ==========================================

direction_counts = (
    df[df["direction"] != "UNKNOWN"]
    .groupby(
        ["track_id", "direction"]
    )
    .size()
    .reset_index(name="count")
)

dominant_direction = (
    direction_counts
    .sort_values(
        ["track_id", "count"],
        ascending=[True, False]
    )
    .drop_duplicates("track_id")
    [["track_id", "direction"]]
)

summary = summary.merge(
    dominant_direction,
    on="track_id",
    how="left"
)


# ==========================================
# 11. SAVE SUMMARY
# ==========================================

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("SMOOTHED TRAJECTORY ANALYSIS")
print("======================================")

print(
    "Valid vehicles:",
    summary["track_id"].nunique()
)

print("\nSample results:\n")

print(
    summary.head(20).to_string(
        index=False
    )
)

print("\n======================================")
print("FILES CREATED")
print("======================================")

print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nTrajectory smoothing completed!")