import cv2
import numpy as np
import pandas as pd
import os
from collections import deque

# =========================================================
# SETTINGS
# =========================================================

VIDEO_FILE = "traffic2.mp4"
TRACK_FILE = "traffic_tracks.csv"

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 675

HISTORY_SECONDS = 10

ZONE_FILES = {
    "ZONE 1": "lane_zone_points.npy",
    "ZONE 2": "zone_2_points.npy",
    "ZONE 3": "zone_3_points.npy",
    "ZONE 4": "zone_4_points.npy",
    "ZONE 5": "zone_5_points.npy",
    "ZONE 6": "zone_6_points.npy",
}

# =========================================================
# LOAD TRACKING DATA
# =========================================================

if not os.path.exists(TRACK_FILE):
    print(f"ERROR: {TRACK_FILE} not found.")
    exit()

df = pd.read_csv(TRACK_FILE)

required_columns = [
    "frame", "track_id", "x1", "y1", "x2", "y2"
]

for column in required_columns:
    if column not in df.columns:
        print(f"ERROR: Missing column: {column}")
        exit()

# Convert frame to integer
df["frame"] = df["frame"].astype(int)

# =========================================================
# LOAD ZONES
# =========================================================

zones = {}

for zone_name, filename in ZONE_FILES.items():

    if os.path.exists(filename):

        points = np.load(filename)

        if len(points) == 4:
            zones[zone_name] = points.astype(np.int32)

print(f"Loaded {len(zones)} zones.")

# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

HISTORY_FRAMES = max(
    1,
    int(HISTORY_SECONDS * fps)
)

# =========================================================
# HISTORY STORAGE
# =========================================================

zone_history = {}

for zone_name in zones:

    zone_history[zone_name] = {
        "counts": deque(maxlen=HISTORY_FRAMES),
        "high_start": None,
        "peak": 0
    }

# =========================================================
# TRAFFIC LEVEL
# =========================================================

def get_level(count):

    if count <= 2:
        return "LOW"

    elif count <= 5:
        return "MEDIUM"

    return "HIGH"


def get_color(level):

    if level == "LOW":
        return (0, 220, 0)

    elif level == "MEDIUM":
        return (0, 200, 255)

    return (0, 0, 255)


# =========================================================
# FRAME DATA LOOKUP
# =========================================================

# Prepare frame groups once.
# This is faster than searching the entire CSV every frame.

frame_groups = {
    int(frame): group
    for frame, group in df.groupby("frame")
}

# =========================================================
# START
# =========================================================

paused = False

print()
print("==========================================")
print(" SMART TRAFFIC MANAGEMENT SYSTEM")
print(" AUTHORITY MODE")
print("==========================================")
print(f"History window: {HISTORY_SECONDS} seconds")
print("SPACE = Pause / Resume")
print("Q     = Quit")
print()

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    if not paused:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        ) - 1

        original_height, original_width = frame.shape[:2]

        sx = DISPLAY_WIDTH / original_width
        sy = DISPLAY_HEIGHT / original_height

        display = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        # =================================================
        # TOP BAR
        # =================================================

        cv2.rectangle(
            display,
            (0, 0),
            (DISPLAY_WIDTH, 55),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display,
            "SMART TRAFFIC MANAGEMENT",
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "AUTHORITY MODE",
            (930, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2
        )

        # =================================================
        # CURRENT FRAME DATA
        # =================================================

        current_data = frame_groups.get(
            frame_number,
            pd.DataFrame()
        )

        zone_results = {}

        # =================================================
        # ANALYZE EACH ZONE
        # =================================================

        for zone_name, polygon in zones.items():

            vehicle_ids = set()

            if not current_data.empty:

                for _, row in current_data.iterrows():

                    center_x = (
                        float(row["x1"]) +
                        float(row["x2"])
                    ) / 2

                    bottom_y = float(row["y2"])

                    inside = cv2.pointPolygonTest(
                        polygon,
                        (center_x, bottom_y),
                        False
                    )

                    if inside >= 0:
                        vehicle_ids.add(row["track_id"])

            current_count = len(vehicle_ids)

            # Add current value to history
            history = zone_history[zone_name]["counts"]
            history.append(current_count)

            # Current status
            current_level = get_level(current_count)

            # Historical statistics
            values = list(history)

            if values:

                average_count = float(
                    np.mean(values)
                )

                peak_count = int(
                    max(values)
                )

            else:

                average_count = 0
                peak_count = 0

            # Historical status
            history_level = get_level(
                round(average_count)
            )

            # Update peak
            zone_history[zone_name]["peak"] = max(
                zone_history[zone_name]["peak"],
                current_count
            )

            # =================================================
            # SUSTAINED HIGH TRAFFIC
            # =================================================

            if history_level == "HIGH":

                if zone_history[zone_name]["high_start"] is None:

                    zone_history[zone_name]["high_start"] = (
                        frame_number / fps
                    )

                sustained_seconds = (
                    frame_number / fps
                    - zone_history[zone_name]["high_start"]
                )

            else:

                zone_history[zone_name]["high_start"] = None
                sustained_seconds = 0

            zone_results[zone_name] = {
                "current": current_count,
                "average": average_count,
                "peak": peak_count,
                "current_level": current_level,
                "history_level": history_level,
                "sustained": sustained_seconds
            }

            # =================================================
            # DRAW ZONE
            # =================================================

            scaled_polygon = np.array(
                [
                    [
                        int(x * sx),
                        int(y * sy)
                    ]
                    for x, y in polygon
                ],
                dtype=np.int32
            )

            color = get_color(history_level)

            # Light transparent zone fill
            overlay = display.copy()

            cv2.fillPoly(
                overlay,
                [scaled_polygon],
                color
            )

            display = cv2.addWeighted(
                overlay,
                0.10,
                display,
                0.90,
                0
            )

            # Zone boundary
            cv2.polylines(
                display,
                [scaled_polygon],
                True,
                color,
                2
            )

            # =================================================
            # SMALL ZONE LABEL
            # =================================================

            label_x = int(
                np.mean(scaled_polygon[:, 0])
            )

            label_y = int(
                np.min(scaled_polygon[:, 1])
            ) + 28

            label_x = max(
                40,
                min(label_x, DISPLAY_WIDTH - 40)
            )

            label_y = max(
                75,
                min(label_y, 600)
            )

            cv2.putText(
                display,
                zone_name.replace("ZONE ", "Z"),
                (label_x - 12, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # =================================================
        # DRAW VEHICLES
        # =================================================

        if not current_data.empty:

            for _, row in current_data.iterrows():

                x1 = int(
                    float(row["x1"]) * sx
                )

                y1 = int(
                    float(row["y1"]) * sy
                )

                x2 = int(
                    float(row["x2"]) * sx
                )

                y2 = int(
                    float(row["y2"]) * sy
                )

                center_x = (
                    float(row["x1"]) +
                    float(row["x2"])
                ) / 2

                bottom_y = float(row["y2"])

                inside_any_zone = False

                for polygon in zones.values():

                    if cv2.pointPolygonTest(
                        polygon,
                        (center_x, bottom_y),
                        False
                    ) >= 0:

                        inside_any_zone = True
                        break

                if inside_any_zone:

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        f"ID {int(row['track_id'])}",
                        (
                            x1,
                            max(y1 - 6, 65)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (255, 255, 255),
                        1
                    )

        # =================================================
        # RIGHT STATUS PANEL
        # =================================================

        panel_x = 925
        panel_y = 70
        panel_w = 260
        panel_h = 500

        cv2.rectangle(
            display,
            (panel_x, panel_y),
            (
                panel_x + panel_w,
                panel_y + panel_h
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display,
            "LIVE ZONE ANALYSIS",
            (panel_x + 15, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        y = panel_y + 60

        # =================================================
        # ZONE STATUS
        # =================================================

        for zone_name, result in zone_results.items():

            level = result["history_level"]
            color = get_color(level)

            cv2.putText(
                display,
                zone_name,
                (panel_x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            cv2.putText(
                display,
                level,
                (panel_x + 135, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2
            )

            y += 21

            cv2.putText(
                display,
                f"Now: {result['current']}",
                (panel_x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (185, 185, 185),
                1
            )

            cv2.putText(
                display,
                f"Avg: {result['average']:.1f}",
                (panel_x + 90, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (185, 185, 185),
                1
            )

            cv2.putText(
                display,
                f"Peak: {result['peak']}",
                (panel_x + 165, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (185, 185, 185),
                1
            )

            y += 34

        # =================================================
        # FIND MOST CONGESTED ZONE
        # =================================================

        if zone_results:

            most_congested_zone = max(
                zone_results,
                key=lambda z:
                zone_results[z]["average"]
            )

            busiest_average = zone_results[
                most_congested_zone
            ]["average"]

            busiest_level = zone_results[
                most_congested_zone
            ]["history_level"]

        else:

            most_congested_zone = "NONE"
            busiest_average = 0
            busiest_level = "LOW"

        # =================================================
        # BOTTOM INFORMATION BAR
        # =================================================

        cv2.rectangle(
            display,
            (0, 625),
            (DISPLAY_WIDTH, 675),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}",
            (20, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 200, 200),
            1
        )

        cv2.putText(
            display,
            f"History: {HISTORY_SECONDS}s",
            (145, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 200, 200),
            1
        )

        cv2.putText(
            display,
            f"Most Congested: {most_congested_zone}",
            (310, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            get_color(busiest_level),
            2
        )

        cv2.putText(
            display,
            f"Avg Vehicles: {busiest_average:.1f}",
            (625, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 200, 200),
            1
        )

        cv2.putText(
            display,
            "Q: Quit | SPACE: Pause",
            (900, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (180, 180, 180),
            1
        )

        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            "Smart Traffic Authority Dashboard",
            display
        )

    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(
        max(1, int(1000 / fps))
    ) & 0xFF

    if key == ord("q"):

        break

    elif key == 32:

        paused = not paused

# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()

print()
print("Dashboard closed.")