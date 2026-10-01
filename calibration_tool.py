import cv2
import numpy as np

VIDEO = "traffic2.mp4"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 560

points = []


# --------------------------------------------------
# LOAD VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO)

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read traffic2.mp4")
    exit()

original_height, original_width = frame.shape[:2]

print("Original video:")
print("Width :", original_width)
print("Height:", original_height)


# --------------------------------------------------
# RESIZE FOR DISPLAY
# --------------------------------------------------

scale_x = original_width / WINDOW_WIDTH
scale_y = original_height / WINDOW_HEIGHT

display_frame = cv2.resize(
    frame,
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)


# --------------------------------------------------
# CREATE SIDE-BY-SIDE DISPLAY
# --------------------------------------------------

def create_display():

    left = display_frame.copy()

    # Draw selected points
    for i, point in enumerate(points):

        x = int(point[0] / scale_x)
        y = int(point[1] / scale_y)

        cv2.circle(
            left,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            left,
            f"P{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Draw lines between points
    if len(points) >= 2:

        for i in range(len(points) - 1):

            x1 = int(points[i][0] / scale_x)
            y1 = int(points[i][1] / scale_y)

            x2 = int(points[i + 1][0] / scale_x)
            y2 = int(points[i + 1][1] / scale_y)

            cv2.line(
                left,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    if len(points) == 4:

        x1 = int(points[3][0] / scale_x)
        y1 = int(points[3][1] / scale_y)

        x2 = int(points[0][0] / scale_x)
        y2 = int(points[0][1] / scale_y)

        cv2.line(
            left,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    # --------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------

    right = np.zeros_like(left)

    if len(points) == 4:

        # Required order:
        # top-left
        # top-right
        # bottom-right
        # bottom-left

        src = np.array([
            points[0],
            points[1],
            points[2],
            points[3]
        ], dtype=np.float32)

        # Output dimensions
        out_w = WINDOW_WIDTH
        out_h = WINDOW_HEIGHT

        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(
            src,
            dst
        )

        right = cv2.warpPerspective(
            display_frame,
            matrix,
            (out_w, out_h)
        )

        cv2.putText(
            right,
            "BIRD'S-EYE VIEW",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            right,
            "Select 4 points",
            (300, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------
    # COMBINE
    # --------------------------------------------------

    combined = np.hstack((left, right))

    return combined


# --------------------------------------------------
# MOUSE CALLBACK
# --------------------------------------------------

def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            original_x = x * scale_x
            original_y = y * scale_y

            points.append(
                (original_x, original_y)
            )

            print(
                f"P{len(points)} = "
                f"({int(original_x)}, "
                f"{int(original_y)})"
            )


# --------------------------------------------------
# WINDOW
# --------------------------------------------------

cv2.namedWindow(
    "Calibration Tool",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Calibration Tool",
    WINDOW_WIDTH * 2,
    WINDOW_HEIGHT
)

cv2.setMouseCallback(
    "Calibration Tool",
    mouse_callback
)


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    output = create_display()

    cv2.imshow(
        "Calibration Tool",
        output
    )

    key = cv2.waitKey(30) & 0xFF

    # ----------------------------------------------
    # R = RESET
    # ----------------------------------------------

    if key == ord("r"):

        points.clear()

        print("\nPoints reset.")

    # ----------------------------------------------
    # S = SAVE
    # ----------------------------------------------

    elif key == ord("s"):

        if len(points) == 4:

            # Save original coordinates
            np.save(
                "calibration_points.npy",
                np.array(points)
            )

            # Calculate perspective matrix
            src = np.array(
                points,
                dtype=np.float32
            )

            dst = np.array([
                [0, 0],
                [WINDOW_WIDTH - 1, 0],
                [WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1],
                [0, WINDOW_HEIGHT - 1]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(
                src,
                dst
            )

            np.save(
                "perspective_matrix.npy",
                matrix
            )

            print("\n================================")
            print("CALIBRATION SAVED")
            print("================================")

            print("Points:")

            for i, p in enumerate(points):
                print(
                    f"P{i + 1}: "
                    f"({int(p[0])}, {int(p[1])})"
                )

            print("\nCreated:")
            print("calibration_points.npy")
            print("perspective_matrix.npy")

            break

        else:

            print(
                "\nPlease select exactly "
                "4 points."
            )

    # ----------------------------------------------
    # ESC = EXIT
    # ----------------------------------------------

    elif key == 27:

        break


cv2.destroyAllWindows()
cap.release()
