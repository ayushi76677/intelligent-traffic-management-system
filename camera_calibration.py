import cv2
import numpy as np

VIDEO = "traffic2.mp4"

points = []

cap = cv2.VideoCapture(VIDEO)

ret, frame = cap.read()

if not ret:
    print("Could not open video")
    exit()

# Original video dimensions
original_height, original_width = frame.shape[:2]

print("Original video:")
print("Width :", original_width)
print("Height:", original_height)

# Window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 675

scale_x = original_width / WINDOW_WIDTH
scale_y = original_height / WINDOW_HEIGHT

display = cv2.resize(
    frame,
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)

def mouse(event, x, y, flags, param):

    global display

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            # Convert displayed coordinates
            # back to original video coordinates
            original_x = int(x * scale_x)
            original_y = int(y * scale_y)

            points.append(
                (original_x, original_y)
            )

            # Draw point on displayed image
            cv2.circle(
                display,
                (x, y),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                f"P{len(points)}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            print(
                f"P{len(points)} = "
                f"({original_x}, {original_y})"
            )


cv2.namedWindow("Calibration")

cv2.setMouseCallback(
    "Calibration",
    mouse
)

while True:

    cv2.imshow(
        "Calibration",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # Reset
    if key == ord("r"):

        points.clear()

        display = cv2.resize(
            frame,
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        print("Points reset.")

    # Save
    elif key == ord("s"):

        if len(points) == 4:

            np.save(
                "calibration_points.npy",
                np.array(points)
            )

            print("\nCalibration saved!")
            print("Original coordinates:")
            print(points)

            break

        else:

            print(
                "Select exactly 4 points first."
            )

    # ESC
    elif key == 27:

        break


cv2.destroyAllWindows()

print("\nFinal points:")
print(points)