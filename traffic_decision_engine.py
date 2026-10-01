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

MEDIUM_AVG = 3
HIGH_AVG = 6

# Number of seconds used to compare
# recent traffic against earlier traffic
TREND_WINDOW_SECONDS = 3

ZONE_FILES = {
    "ZONE 1": "lane_zone_points.npy",
    "ZONE 2": "zone_2_points.npy",
    "ZONE 3": "zone_3_points.npy",
    "ZONE 4": "zone_4_points.npy",
    "ZONE 5": "zone_5_points.npy",
    "ZONE 6": "zone_6_points.npy",
}

# =========================================================
# LOAD DATA
# =========================================================

if not os.path.exists(TRACK_FILE):
    print(f"ERROR: {TRACK_FILE} not found.")
    raise SystemExit

df = pd.read_csv(TRACK_FILE)

required_columns = [
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2"
]

for column in required_columns:

    if column not in df.columns:

        print(
            f"ERROR: Missing column: {column}"
        )

        raise SystemExit

df["frame"] = df["frame"].astype(int)

# =========================================================
# LOAD ZONES
# =========================================================

zones = {}

for zone_name, filename in ZONE_FILES.items():

    if not os.path.exists(filename):

        print(
            f"WARNING: {filename} not found."
        )

        continue

    points = np.load(filename)

    if len(points) == 4:

        zones[zone_name] = (
            points.astype(np.int32)
        )

print(
    f"Loaded {len(zones)} zones."
)

if not zones:

    print("ERROR: No zones found.")
    raise SystemExit

# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(
    VIDEO_FILE
)

if not cap.isOpened():

    print(
        f"ERROR: Could not open {VIDEO_FILE}"
    )

    raise SystemExit

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30

HISTORY_FRAMES = max(
    1,
    int(HISTORY_SECONDS * fps)
)

TREND_FRAMES = max(
    1,
    int(TREND_WINDOW_SECONDS * fps)
)

# =========================================================
# GROUP DATA BY FRAME
# =========================================================

frame_groups = {
    int(frame): group
    for frame, group
    in df.groupby("frame")
}

# =========================================================
# HISTORY
# =========================================================

zone_history = {}

for zone_name in zones:

    zone_history[zone_name] = deque(
        maxlen=HISTORY_FRAMES
    )

# =========================================================
# FUNCTIONS
# =========================================================

def classify_zone(avg_count):

    if avg_count < MEDIUM_AVG:
        return "LOW"

    elif avg_count < HIGH_AVG:
        return "MEDIUM"

    return "HIGH"


def color_for_level(level):

    if level == "LOW":
        return (0, 220, 0)

    elif level == "MEDIUM":
        return (0, 200, 255)

    return (0, 0, 255)


def calculate_trend(history):

    values = list(history)

    if len(values) < TREND_FRAMES * 2:

        return "STABLE"

    recent = np.mean(
        values[-TREND_FRAMES:]
    )

    previous = np.mean(
        values[-TREND_FRAMES * 2:
               -TREND_FRAMES]
    )

    difference = recent - previous

    # Small changes are treated as stable
    tolerance = 0.75

    if difference > tolerance:

        return "INCREASING"

    elif difference < -tolerance:

        return "DECREASING"

    return "STABLE"


def trend_color(trend):

    if trend == "INCREASING":
        return (0, 0, 255)

    elif trend == "DECREASING":
        return (0, 220, 0)

    return (0, 200, 255)


def get_recommendation(
    level,
    trend,
    sustained
):

    if level == "HIGH":

        if trend == "INCREASING":

            return "CONGESTION BUILDING"

        elif trend == "DECREASING":

            return "CONGESTION CLEARING"

        else:

            return "SUSTAINED HIGH TRAFFIC"

    if level == "MEDIUM":

        if trend == "INCREASING":

            return "MONITOR - BUILDING"

        elif trend == "DECREASING":

            return "MONITOR - CLEARING"

        else:

            return "MONITOR"

    return "NORMAL FLOW"


def find_priority_zone(results):

    if not results:
        return None

    level_score = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2
    }

    trend_score = {
        "DECREASING": 0,
        "STABLE": 1,
        "INCREASING": 2
    }

    return max(
        results.keys(),
        key=lambda zone: (

            level_score[
                results[zone]["level"]
            ],

            trend_score[
                results[zone]["trend"]
            ],

            results[zone]["average"],

            results[zone]["sustained"]
        )
    )


# =========================================================
# START
# =========================================================

paused = False

print()
print(
    "=========================================="
)

print(
    " SMART TRAFFIC DECISION ENGINE"
)

print(
    "=========================================="
)

print(
    f"History window: {HISTORY_SECONDS}s"
)

print(
    f"Trend window: {TREND_WINDOW_SECONDS}s"
)

print(
    "SPACE = Pause / Resume"
)

print(
    "Q = Quit"
)

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
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        ) - 1

        original_height, original_width = (
            frame.shape[:2]
        )

        sx = (
            DISPLAY_WIDTH /
            original_width
        )

        sy = (
            DISPLAY_HEIGHT /
            original_height
        )

        display = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )

        # =================================================
        # HEADER
        # =================================================

        cv2.rectangle(
            display,
            (0, 0),
            (
                DISPLAY_WIDTH,
                55
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display,
            "SMART TRAFFIC DECISION ENGINE",
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "AUTHORITY MODE",
            (970, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 200, 255),
            2
        )

        # =================================================
        # CURRENT DATA
        # =================================================

        current_data = frame_groups.get(
            frame_number,
            pd.DataFrame()
        )

        results = {}

        # =================================================
        # PROCESS ZONES
        # =================================================

        for zone_name, polygon in zones.items():

            vehicle_ids = set()

            if not current_data.empty:

                for _, row in current_data.iterrows():

                    center_x = (
                        float(row["x1"]) +
                        float(row["x2"])
                    ) / 2

                    bottom_y = float(
                        row["y2"]
                    )

                    inside = (
                        cv2.pointPolygonTest(
                            polygon,
                            (
                                center_x,
                                bottom_y
                            ),
                            False
                        )
                    )

                    if inside >= 0:

                        vehicle_ids.add(
                            row["track_id"]
                        )

            current_count = len(
                vehicle_ids
            )

            # Store history
            zone_history[
                zone_name
            ].append(
                current_count
            )

            history_values = list(
                zone_history[
                    zone_name
                ]
            )

            if history_values:

                average_count = float(
                    np.mean(
                        history_values
                    )
                )

                peak_count = int(
                    max(
                        history_values
                    )
                )

            else:

                average_count = 0.0
                peak_count = 0

            # Classification
            level = classify_zone(
                average_count
            )

            # Trend
            trend = calculate_trend(
                zone_history[
                    zone_name
                ]
            )

            # =================================================
            # SUSTAINED HIGH TRAFFIC
            # =================================================

            sustained_seconds = 0.0

            if level == "HIGH":

                for value in reversed(
                    history_values
                ):

                    if classify_zone(
                        value
                    ) == "HIGH":

                        sustained_seconds += (
                            1 / fps
                        )

                    else:

                        break

            recommendation_text = (
                get_recommendation(
                    level,
                    trend,
                    sustained_seconds
                )
            )

            results[zone_name] = {

                "current":
                    current_count,

                "average":
                    average_count,

                "peak":
                    peak_count,

                "level":
                    level,

                "trend":
                    trend,

                "sustained":
                    sustained_seconds,

                "recommendation":
                    recommendation_text
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

            color = color_for_level(
                level
            )

            overlay = display.copy()

            cv2.fillPoly(
                overlay,
                [scaled_polygon],
                color
            )

            display = cv2.addWeighted(
                overlay,
                0.08,
                display,
                0.92,
                0
            )

            cv2.polylines(
                display,
                [scaled_polygon],
                True,
                color,
                2
            )

            # Zone label
            label_x = int(
                np.mean(
                    scaled_polygon[:, 0]
                )
            )

            label_y = int(
                np.mean(
                    scaled_polygon[:, 1]
                )
            )

            label_x = max(
                35,
                min(
                    label_x,
                    DISPLAY_WIDTH - 35
                )
            )

            label_y = max(
                75,
                min(
                    label_y,
                    610
                )
            )

            cv2.putText(
                display,
                zone_name.replace(
                    "ZONE ",
                    "Z"
                ),
                (
                    label_x - 12,
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
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

                bottom_y = float(
                    row["y2"]
                )

                inside_any_zone = False

                for polygon in zones.values():

                    if cv2.pointPolygonTest(
                        polygon,
                        (
                            center_x,
                            bottom_y
                        ),
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
                            max(
                                y1 - 5,
                                65
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (255, 255, 255),
                        1
                    )

        # =================================================
        # PRIORITY ZONE
        # =================================================

        priority_zone = (
            find_priority_zone(
                results
            )
        )

        # =================================================
        # RIGHT PANEL
        # =================================================

        panel_x = 900
        panel_y = 70
        panel_w = 290
        panel_h = 505

        cv2.rectangle(
            display,
            (
                panel_x,
                panel_y
            ),
            (
                panel_x + panel_w,
                panel_y + panel_h
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display,
            "TRAFFIC INTELLIGENCE",
            (
                panel_x + 15,
                panel_y + 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y = panel_y + 60

        # =================================================
        # ZONE INFORMATION
        # =================================================

        for zone_name, result in results.items():

            level = result["level"]
            trend = result["trend"]

            level_color = color_for_level(
                level
            )

            trend_col = trend_color(
                trend
            )

            cv2.putText(
                display,
                zone_name,
                (
                    panel_x + 15,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1
            )

            cv2.putText(
                display,
                level,
                (
                    panel_x + 135,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                level_color,
                2
            )

            y += 18

            cv2.putText(
                display,
                f"Now {result['current']}",
                (
                    panel_x + 15,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1
            )

            cv2.putText(
                display,
                f"Avg {result['average']:.1f}",
                (
                    panel_x + 90,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1
            )

            y += 18

            cv2.putText(
                display,
                "Trend:",
                (
                    panel_x + 15,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (170, 170, 170),
                1
            )

            cv2.putText(
                display,
                trend,
                (
                    panel_x + 75,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                trend_col,
                1
            )

            y += 30

        # =================================================
        # PRIORITY BOX
        # =================================================

        if priority_zone:

            priority = results[
                priority_zone
            ]

            priority_level_color = (
                color_for_level(
                    priority["level"]
                )
            )

            box_top = 410

            cv2.rectangle(
                display,
                (
                    panel_x + 10,
                    box_top
                ),
                (
                    panel_x + panel_w - 10,
                    box_top + 145
                ),
                (35, 35, 35),
                -1
            )

            cv2.putText(
                display,
                "PRIORITY ZONE",
                (
                    panel_x + 20,
                    box_top + 23
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (170, 170, 170),
                1
            )

            cv2.putText(
                display,
                priority_zone,
                (
                    panel_x + 20,
                    box_top + 52
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                priority_level_color,
                2
            )

            cv2.putText(
                display,
                priority["level"],
                (
                    panel_x + 150,
                    box_top + 52
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                priority_level_color,
                2
            )

            cv2.putText(
                display,
                f"Trend: {priority['trend']}",
                (
                    panel_x + 20,
                    box_top + 79
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                trend_color(
                    priority["trend"]
                ),
                1
            )

            cv2.putText(
                display,
                (
                    f"Average: "
                    f"{priority['average']:.1f}"
                ),
                (
                    panel_x + 20,
                    box_top + 102
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (210, 210, 210),
                1
            )

            cv2.putText(
                display,
                priority[
                    "recommendation"
                ],
                (
                    panel_x + 20,
                    box_top + 128
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                priority_level_color,
                1
            )

        # =================================================
        # BOTTOM BAR
        # =================================================

        cv2.rectangle(
            display,
            (0, 625),
            (
                DISPLAY_WIDTH,
                675
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}",
            (20, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (190, 190, 190),
            1
        )

        cv2.putText(
            display,
            f"History: {HISTORY_SECONDS}s",
            (150, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (190, 190, 190),
            1
        )

        if priority_zone:

            cv2.putText(
                display,
                f"Attention: {priority_zone}",
                (330, 655),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color_for_level(
                    results[
                        priority_zone
                    ]["level"]
                ),
                2
            )

        cv2.putText(
            display,
            "SPACE: Pause | Q: Quit",
            (950, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1
        )

        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            "Smart Traffic Decision Engine",
            display
        )

    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(
        max(
            1,
            int(1000 / fps)
        )
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
print(
    "Traffic Decision Engine closed."
)