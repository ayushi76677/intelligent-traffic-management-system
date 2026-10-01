import cv2
import numpy as np

VIDEO = "traffic2.mp4"
POINTS_FILE = "calibration_points.npy"

# -----------------------------------------
# 1. Load calibration points
# -----------------------------------------

points = np.load(POINTS_FILE).astype(np.float32)

print("Original points:")
print(points)

# Your saved order is:
# P1 = top-left
# P2 = top-right
# P3 = bottom-left
# P4 = bottom-right
#
# OpenCV needs:
# top-left
# top-right
# bottom-right
# bottom-left

src = np.array([
    points[0],   # P1
    points[1],   # P2
    points[3],   # P4
    points[2]    # P3
], dtype=np.float32)


# -----------------------------------------
# 2. Output bird's-eye dimensions
# -----------------------------------------

WIDTH = 1200
HEIGHT = 700

dst = np.array([
    [0, 0],
    [WIDTH - 1, 0],
    [WIDTH - 1, HEIGHT - 1],
    [0, HEIGHT - 1]
], dtype=np.float32)


# -----------------------------------------
# 3. Calculate perspective matrix
# -----------------------------------------

matrix = cv2.getPerspectiveTransform(
    src,
    dst
)

np.save(
    "perspective_matrix.npy",
    matrix
)

print("\nPerspective matrix saved!")


# -----------------------------------------
# 4. Read first video frame
# -----------------------------------------

cap = cv2.VideoCapture(VIDEO)

ret, frame = cap.read()

if not ret:
    print("Could not read video.")
    exit()


# -----------------------------------------
# 5. Transform frame
# -----------------------------------------

bird_view = cv2.warpPerspective(
    frame,
    matrix,
    (WIDTH, HEIGHT)
)


# -----------------------------------------
# 6. Save bird's-eye image
# -----------------------------------------

cv2.imwrite(
    "bird_eye_view.jpg",
    bird_view
)

print("Bird's-eye view saved:")
print("bird_eye_view.jpg")


# -----------------------------------------
# 7. Display
# -----------------------------------------

cv2.imshow(
    "Bird's-Eye View",
    bird_view
)

print("\nPress any key to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()
cap.release()