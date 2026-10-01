import cv2
import numpy as np
import os

VIDEO = "traffic2.mp4"

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 675
MAX_POINTS = 4

points = []

# Find the next zone number automatically
zone_number = 2
while os.path.exists(f"zone_{zone_number}_points.npy"):
    zone_number += 1


def redraw(frame, points):
    display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Draw points
    for i, (x, y) in enumerate(points):
        dx = int(x * DISPLAY_WIDTH / frame.shape[1])
        dy = int(y * DISPLAY_HEIGHT / frame.shape[0])

        cv2.circle(display, (dx, dy), 6, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"P{i+1}",
            (dx + 10, dy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Draw polygon when 4 points exist
    if len(points) >= 2:
        scaled_points = [
            (
                int(x * DISPLAY_WIDTH / frame.shape[1]),
                int(y * DISPLAY_HEIGHT / frame.shape[0])
            )
            for x, y in points
        ]

        cv2.polylines(
            display,
            [np.array(scaled_points)],
            len(points) == 4,
            (0, 255, 0),
            2
        )

    # Instructions
    cv2.putText(
        display,
        f"ZONE {zone_number} | Click 4 points | R=Reset | S=Save | ESC=Exit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return display


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < MAX_POINTS:

            original_x = int(x * frame.shape[1] / DISPLAY_WIDTH)
            original_y = int(y * frame.shape[0] / DISPLAY_HEIGHT)

            points.append((original_x, original_y))

            print(
                f"P{len(points)} = ({original_x}, {original_y})"
            )


cap = cv2.VideoCapture(VIDEO)

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read video.")
    cap.release()
    exit()

cv2.namedWindow("Zone Selector")
cv2.setMouseCallback("Zone Selector", mouse_callback)

while True:

    display = redraw(frame, points)

    cv2.imshow("Zone Selector", display)

    key = cv2.waitKey(30) & 0xFF

    # RESET
    if key == ord("r"):

        points.clear()

        print("\nPoints reset.")
        print(f"Select 4 points for ZONE {zone_number} again.")

    # SAVE
    elif key == ord("s"):

        if len(points) != 4:

            print(
                f"\nYou selected {len(points)} points."
                "\nYou must select exactly 4 points."
            )

            continue

        filename = f"zone_{zone_number}_points.npy"

        np.save(filename, np.array(points, dtype=np.int32))

        print("\n--------------------------------")
        print(f"ZONE {zone_number} SAVED")
        print("--------------------------------")
        print(f"File: {filename}")
        print("Coordinates:")

        for i, p in enumerate(points):
            print(f"P{i+1} = {tuple(p)}")

        print("--------------------------------")

        # Move to next zone
        zone_number += 1

        while os.path.exists(f"zone_{zone_number}_points.npy"):
            zone_number += 1

        points.clear()

        print(f"\nNow select 4 points for ZONE {zone_number}.")

    # EXIT
    elif key == 27:
        break


cap.release()
cv2.destroyAllWindows()