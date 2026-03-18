from pathlib import Path as FSPath
from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Set path to dataset and runs directory
    print("Dataset:", FSPath("D:/yolo_dataset/dataset.yaml").resolve())
    print("Runs dir:", FSPath("D:/yolo_runs").resolve())

    # Train on Mapillary dataset
    # Epochs, imgsz, batch size downshifted for quick testing
    results = model.train(
        data="D:/yolo_dataset/dataset.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        workers=0,
        project="D:/yolo_runs",
        name="recovery_run",
        pretrained=True,
        save=True,
        plots=True,
        verbose=True,
    )

    print(results)

if __name__ == "__main__":
    main()