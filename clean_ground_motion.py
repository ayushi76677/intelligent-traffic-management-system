import pandas as pd
import numpy as np

INPUT_CSV = "vehicle_ground_plane.csv"
OUTPUT_CSV = "clean_ground_motion.csv"

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

SMOOTHING_WINDOW = 7

# Maximum allowed ground-plane displacement between
# consecutive observations before treating it as noise.
MAX_JUMP = 10.0

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)

print("Loaded observations:", len(df))


# ------------------------------------------------------------
# SMOOTH GROUND POSITION
# ------------------------------------------------------------

df["smooth_ground_x"] = (
    df.groupby("track_id")["ground_x"]
    .transform(
        lambda x: x.rolling(
            SMOOTHING_WINDOW,
            min_periods=1,
            center=True
        ).median()
    )
)

df["smooth_ground_y"] = (
    df.groupby("track_id")["ground_y"]
    .transform(
        lambda x: x.rolling(
            SMOOTHING_WINDOW,
            min_periods=1,
            center=True
        ).median()
    )
)


# ------------------------------------------------------------
# CALCULATE CHANGE
# ------------------------------------------------------------

df["smooth_dx"] = (
    df.groupby("track_id")["smooth_ground_x"]
    .diff()
)

df["smooth_dy"] = (
    df.groupby("track_id")["smooth_ground_y"]
    .diff()
)

df["smooth_distance"] = np.sqrt(
    df["smooth_dx"] ** 2 +
    df["smooth_dy"] ** 2
)


# ------------------------------------------------------------
# REMOVE IMPOSSIBLE JUMPS
# ------------------------------------------------------------

df["valid_motion"] = (
    df["smooth_distance"] <= MAX_JUMP
)

# First observation of every vehicle is valid
first_observation = (
    df.groupby("track_id")
    .cumcount() == 0
)

df.loc[
    first_observation,
    "valid_motion"
] = True


# ------------------------------------------------------------
# CLEAN DISTANCE
# ------------------------------------------------------------

df["clean_distance"] = np.where(
    df["valid_motion"],
    df["smooth_distance"],
    0
)


# ------------------------------------------------------------
# TIME DIFFERENCE
# ------------------------------------------------------------

df["time_difference"] = (
    df.groupby("track_id")["time"]
    .diff()
)


# ------------------------------------------------------------
# CLEAN SPEED
# ------------------------------------------------------------

df["clean_speed"] = np.where(
    (
        (df["time_difference"] > 0) &
        (df["valid_motion"])
    ),
    df["clean_distance"] /
    df["time_difference"],
    0
)


# ------------------------------------------------------------
# DIRECTION
# ------------------------------------------------------------

df["direction"] = "STATIONARY"

moving = df["clean_distance"] > 0.05

df.loc[
    moving & (df["smooth_dx"] > 0),
    "direction"
] = "RIGHT"

df.loc[
    moving & (df["smooth_dx"] < 0),
    "direction"
] = "LEFT"


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n========================================")
print("CLEAN GROUND MOTION COMPLETE")
print("========================================")

print("Output:", OUTPUT_CSV)

print(
    "Valid observations:",
    df["valid_motion"].sum()
)

print(
    "Rejected observations:",
    (~df["valid_motion"]).sum()
)

print("\nExample:\n")

print(
    df[
        [
            "frame",
            "time",
            "track_id",
            "vehicle_type",
            "smooth_ground_x",
            "smooth_ground_y",
            "clean_distance",
            "clean_speed",
            "direction",
            "valid_motion"
        ]
    ]
    .head(30)
    .to_string(index=False)
)

print("\nDone.")