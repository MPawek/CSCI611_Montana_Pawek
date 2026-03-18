from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")  # pre-trained baseline

test_folder = Path("D:/yolo_dataset/images/test")
output_folder = Path("results/baseline")
output_folder.mkdir(parents=True, exist_ok=True)

for img_path in test_folder.glob("*.jpg"):
    results = model(img_path, imgsz=640, conf=0.25)
    results[0].save(filename=str(output_folder / img_path.name))

