from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.track(
    source="traffic2.mp4",
    save=True,
    conf=0.25,
    imgsz=1920,
    classes=[2, 3, 5, 7],
    tracker="bytetrack.yaml",
    vid_stride=1
)

print("Tracking completed!")                