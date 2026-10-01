from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.predict(
    source="traffic2.mp4",
    save=True,
    conf=0.20,
    imgsz=1920,
    classes=[2, 3, 5, 7]
)

print("Detection completed!")