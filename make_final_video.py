import cv2

# Working tracking video
input_video = r"runs\detect\track-3\traffic2.avi"

# Final output
output_video = "traffic_final.avi"

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open tracking video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Frames:", total_frames)
print("Duration:", total_frames / fps, "seconds")

# Create output
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit()

frame_number = 0

# Your previously calculated timeline results
timeline = [
    (0, 10, 6, 11, 2, 4, 23, "MEDIUM"),
    (10, 20, 11, 14, 1, 3, 29, "HIGH"),
    (20, 30, 12, 10, 3, 4, 29, "HIGH"),
    (30, 40, 10, 2, 1, 3, 16, "MEDIUM"),
    (40, 51, 7, 11, 3, 4, 25, "HIGH")
]

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    current_time = frame_number / fps

    # Find current timeline interval
    cars = motorcycles = buses = trucks = total = 0
    traffic = "UNKNOWN"

    for start, end, c, m, b, t, total_count, level in timeline:

        if start <= current_time < end:

            cars = c
            motorcycles = m
            buses = b
            trucks = t
            total = total_count
            traffic = level

            break

    # Add black information panel
    cv2.rectangle(
        frame,
        (20, 20),
        (470, 235),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "TRAFFIC ANALYSIS",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {cars}",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycles}",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {buses}",
        (40, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {trucks}",
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total: {total}  Traffic: {traffic}",
        (40, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Write frame
    out.write(frame)

    if frame_number % 150 == 0:
        print(
            f"Processed {current_time:.1f} seconds "
            f"({frame_number}/{total_frames})"
        )

# Release
cap.release()
out.release()

print()
print("==============================")
print("FINAL VIDEO CREATED")
print("==============================")
print("Output:", output_video)
print("Frames written:", frame_number)
print("Expected frames:", total_frames)
print("Duration:", frame_number / fps, "seconds")