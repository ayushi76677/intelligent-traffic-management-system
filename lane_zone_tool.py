import cv2
import numpy as np

VIDEO = "traffic2.mp4"

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 675

MAX_POINTS = 4

points = []

cap = cv2.VideoCapture(VIDEO)

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not open traffic2.mp4")
    exit()

original_height, original_width = frame.shape[:2]

print("Original video:")
print("Width :", original_width)
print("Height:", original_height)

# Scale original -> display
scale_x = original_width / DISPLAY_WIDTH
scale_y = original_height / DISPLAY_HEIGHT

display = cv2.resize(
    frame,
    (DISPLAY_WIDTH, DISPLAY_HEIGHT)
)


# ---------------------------------------------------------
# MOUSE CALLBACK
# ---------------------------------------------------------

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < MAX_POINTS:

            original_x = int(x * scale_x)
            original_y = int(y * scale_y)

            points.append(
                (original_x, original_y)
            )

            print(
                f"P{len(points)} = "
                f"({original_x}, {original_y})"
            )


# ---------------------------------------------------------
# DRAW
# ---------------------------------------------------------

def create_display():

    img = display.copy()

    display_points = []

    for p in points:

        x = int(p[0] / scale_x)
        y = int(p[1] / scale_y)

        display_points.append((x, y))

        cv2.circle(
            img,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

    # Draw connecting lines
    if len(display_points) >= 2:

        for i in range(len(display_points) - 1):

            cv2.line(
                img,
                display_points[i],
                display_points[i + 1],
                (0, 255, 0),
                2
            )

    # Close polygon
    if len(display_points) == 4:

        cv2.line(
            img,
            display_points[3],
            display_points[0],
            (0, 255, 0),
            2
        )

        # Transparent fill
        overlay = img.copy()

        polygon = np.array(
            display_points,
            dtype=np.int32
        )

        cv2.fillPoly(
            overlay,
            [polygon],
            (255, 255, 255)
        )

        img = cv2.addWeighted(
            overlay,
            0.20,
            img,
            0.80,
            0
        )

    cv2.putText(
        img,
        "Select 4 points | R = reset | S = save",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return img


# ---------------------------------------------------------
# WINDOW
# ---------------------------------------------------------

cv2.namedWindow(
    "Lane Zone Tool",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Lane Zone Tool",
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)

cv2.setMouseCallback(
    "Lane Zone Tool",
    mouse_callback
)


# ---------------------------------------------------------
# LOOP
# ---------------------------------------------------------

while True:

    img = create_display()

    cv2.imshow(
        "Lane Zone Tool",
        img
    )

    key = cv2.waitKey(30) & 0xFF

    # Reset
    if key == ord("r"):

        points.clear()

        print("\nPoints reset.")

    # Save
    elif key == ord("s"):

        if len(points) == 4:

            np.save(
                "lane_zone_points.npy",
                np.array(points)
            )

            print("\n==============================")
            print("ZONE SAVED")
            print("==============================")

            for i, p in enumerate(points):
                print(
                    f"P{i + 1}: "
                    f"({p[0]}, {p[1]})"
                )

            print(
                "\nCreated: "
                "lane_zone_points.npy"
            )

            break

        else:

            print(
                "\nPlease select exactly 4 points."
            )

    elif key == 27:
        break


cv2.destroyAllWindows()
cap.release()