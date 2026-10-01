import cv2
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "traffic_smoothed_movement.csv"
OUTPUT_CSV = "vehicle_ground_plane.csv"

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Your calibrated road points
# Order:
# P1 = top-left
# P2 = top-right
# P3 = bottom-right
# P4 = bottom-left

SOURCE_POINTS = np.array([
    [600, 516],
    [1904, 528],
    [1905, 1056],
    [12, 1062]
], dtype=np.float32)

# Ground-plane coordinate system.
# We use an arbitrary normalized coordinate system for now.
DEST_POINTS = np.array([
    [0, 0],
    [100, 0],
    [100, 50],
    [0, 50]
], dtype=np.float32)


# ============================================================
# CALCULATE HOMOGRAPHY
# ============================================================

H = cv2.getPerspectiveTransform(
    SOURCE_POINTS,
    DEST_POINTS
)

print("Perspective matrix:")
print(H)


# ============================================================
# LOAD TRACKING DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("\nLoaded:", len(df), "observations")


# ============================================================
# VEHICLE REFERENCE POINT
# ============================================================
#
# We use the bottom-center of the bounding box.
#
# This is more appropriate than using the center because
# the bottom of the bounding box approximately represents
# where the vehicle touches the road.
# ============================================================

df["bottom_center_x"] = (
    df["x1"] + df["x2"]
) / 2

df["bottom_center_y"] = df["y2"]


# ============================================================
# TRANSFORM POINTS
# ============================================================

image_points = np.column_stack([
    df["bottom_center_x"].values,
    df["bottom_center_y"].values
]).astype(np.float32)

image_points = image_points.reshape(
    -1, 1, 2
)

ground_points = cv2.perspectiveTransform(
    image_points,
    H
)

ground_points = ground_points.reshape(
    -1, 2
)


df["ground_x"] = ground_points[:, 0]
df["ground_y"] = ground_points[:, 1]


# ============================================================
# CALCULATE MOVEMENT IN GROUND PLANE
# ============================================================

df["ground_dx"] = (
    df.groupby("track_id")["ground_x"]
    .diff()
)

df["ground_dy"] = (
    df.groupby("track_id")["ground_y"]
    .diff()
)


df["ground_distance"] = np.sqrt(
    df["ground_dx"] ** 2 +
    df["ground_dy"] ** 2
)


# ============================================================
# TIME DIFFERENCE
# ============================================================

df["time_difference"] = (
    df.groupby("track_id")["time"]
    .diff()
)


# ============================================================
# GROUND-PLANE SPEED
# ============================================================

df["ground_speed"] = (
    df["ground_distance"] /
    df["time_difference"]
)

df["ground_speed"] = (
    df["ground_speed"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("GROUND-PLANE TRANSFORMATION COMPLETE")
print("============================================")

print("\nOutput file:")
print(OUTPUT_CSV)

print("\nExample results:\n")

print(
    df[
        [
            "frame",
            "time",
            "track_id",
            "vehicle_type",
            "bottom_center_x",
            "bottom_center_y",
            "ground_x",
            "ground_y",
            "ground_speed"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\nDone.")
