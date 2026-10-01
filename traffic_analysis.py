from ultralytics import YOLO
import cv2
import csv

# Load model
model = YOLO("yolov8m.pt")

video_path = "traffic2.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video duration: {duration:.1f} seconds")
print(f"FPS: {fps}")

# Timeline intervals

intervals = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 51)
]

results_data = []

# Analyze one frame from
# the middle of each interval
for start, end in intervals:

    middle_time = (start + end) / 2
    frame_number = int(middle_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        print(f"Could not read frame at {middle_time:.1f}s")
        continue

    # YOLO detection
    results = model.predict(
        frame,
        conf=0.25,
        imgsz=1920,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0

    if results[0].boxes is not None:

        for box in results[0].boxes:

            cls = int(box.cls[0])

            if cls == 2:
                cars += 1

            elif cls == 3:
                motorcycles += 1

            elif cls == 5:
                buses += 1

            elif cls == 7:
                trucks += 1

    total = cars + motorcycles + buses + trucks

    # Traffic level
    if total < 10:
        traffic = "LOW"
    elif total < 25:
        traffic = "MEDIUM"
    else:
        traffic = "HIGH"

    results_data.append([
        f"{start}-{end}",
        cars,
        motorcycles,
        buses,
        trucks,
        total,
        traffic
    ])

    print(
        f"{start}-{end}s | "
        f"Cars: {cars} | "
        f"Motorcycles: {motorcycles} | "
        f"Buses: {buses} | "
        f"Trucks: {trucks} | "
        f"Total: {total} | "
        f"Traffic: {traffic}"
    )

cap.release()
# Save CSV
with open(
    "traffic_timeline.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Time Interval",
        "Cars",
        "Motorcycles",
        "Buses",
        "Trucks",
        "Total Vehicles",
        "Traffic Level"
    ])

    writer.writerows(results_data)

print("\n==============================")
print("TRAFFIC ANALYSIS COMPLETED")
print("==============================")
print("Results saved to: traffic_timeline.csv")