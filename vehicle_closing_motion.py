import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "clean_ground_motion.csv"
OUTPUT_FILE = "vehicle_closing_motion.csv"

# Number of observations used to measure longer-term motion
WINDOW = 10

# Minimum ground-plane movement to consider meaningful
MIN_MOVEMENT = 0.20

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)

print("Loaded observations:", len(df))


# ============================================================
# CALCULATE LONG-TERM MOVEMENT
# ============================================================

df["past_x"] = (
    df.groupby("track_id")["smooth_ground_x"]
    .shift(WINDOW)
)

df["past_y"] = (
    df.groupby("track_id")["smooth_ground_y"]
    .shift(WINDOW)
)

df["past_time"] = (
    df.groupby("track_id")["time"]
    .shift(WINDOW)
)


# ============================================================
# CHANGE IN POSITION
# ============================================================

df["delta_x"] = (
    df["smooth_ground_x"] - df["past_x"]
)

df["delta_y"] = (
    df["smooth_ground_y"] - df["past_y"]
)

df["elapsed_time"] = (
    df["time"] - df["past_time"]
)


df["distance"] = np.sqrt(
    df["delta_x"] ** 2 +
    df["delta_y"] ** 2
)


# ============================================================
# LONG-TERM SPEED
# ============================================================

df["stable_speed"] = np.where(
    df["elapsed_time"] > 0,
    df["distance"] / df["elapsed_time"],
    0
)


# ============================================================
# DETERMINE APPROACHING MOTION
# ============================================================

# In this camera setup, increasing ground_y represents
# movement toward the camera.

df["approaching"] = (
    (df["delta_y"] > MIN_MOVEMENT) &
    (df["distance"] > MIN_MOVEMENT)
)


# ============================================================
# DETERMINE RECEDING MOTION
# ============================================================

df["receding"] = (
    (df["delta_y"] < -MIN_MOVEMENT) &
    (df["distance"] > MIN_MOVEMENT)
)


# ============================================================
# MOTION CLASSIFICATION
# ============================================================

df["motion_state"] = "STATIONARY"

df.loc[
    df["approaching"],
    "motion_state"
] = "APPROACHING"

df.loc[
    df["receding"],
    "motion_state"
] = "RECEDING"


# ============================================================
# APPROACHING SPEED
# ============================================================

df["closing_speed"] = np.where(
    df["approaching"],
    abs(df["delta_y"]) / df["elapsed_time"],
    0
)

df["closing_speed"] = df["closing_speed"].fillna(0)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("VEHICLE CLOSING MOTION COMPLETE")
print("============================================")

print("Output:", OUTPUT_FILE)

print("\nMotion state counts:")

print(
    df["motion_state"]
    .value_counts()
)

print("\nExample results:\n")

print(
    df[
        [
            "frame",
            "time",
            "track_id",
            "vehicle_type",
            "smooth_ground_x",
            "smooth_ground_y",
            "delta_x",
            "delta_y",
            "stable_speed",
            "closing_speed",
            "motion_state"
        ]
    ]
    .head(40)
    .to_string(index=False)
)

print("\nDone.")