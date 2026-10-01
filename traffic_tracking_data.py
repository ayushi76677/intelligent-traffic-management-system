from ultralytics import YOLO
import csv

# Load YOLO model
model = YOLO("yolov8m.pt")

# Open CSV file for saving tracking data
with open("traffic_tracks.csv", "w", newline="") as file:

    writer = csv.writer(file)

    # CSV column names
    writer.writerow([
        "frame",
        "time",
        "track_id",
        "class",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y"
    ])

    # Run YOLO tracking
    results = model.track(
        source="traffic2.mp4",
        conf=0.25,
        imgsz=1920,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml",
        vid_stride=1,
        stream=True
    )

    # Process every frame
    for frame_number, result in enumerate(results):

        # Get video FPS
        fps = result.speed.get("fps", 30)

        # If FPS isn't available, use 30 FPS
        if not fps or fps <= 0:
            fps = 30

        time_seconds = frame_number / fps

        # Check whether tracking IDs exist
        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            # Process every detected vehicle
            for box, track_id, class_id in zip(
                boxes, track_ids, classes
            ):

                x1, y1, x2, y2 = box

                # Calculate center of bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                writer.writerow([
                    frame_number,
                    round(time_seconds, 3),
                    track_id,
                    class_id,
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                    round(center_x, 2),
                    round(center_y, 2)
                ])

print("Tracking data saved successfully!")
print("Output file: traffic_tracks.csv")