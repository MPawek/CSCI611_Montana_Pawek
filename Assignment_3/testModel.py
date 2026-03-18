from ultralytics import YOLO
from pathlib import Path
import csv

if __name__ == "__main__":

    # Path to best performing model
    model = YOLO("D:/yolo_runs/recovery_run/weights/best.pt")
    test_folder = Path("D:/yolo_dataset/images/test")
    output_folder = Path("results/trained")

    # Evaluate on test set, make sure to set verbose to false to avoid printing all the per-image results to the console
    metrics = model.val(data="D:/yolo_dataset/dataset.yaml", split="test", imgsz=640, conf=0.1, verbose=False)

    # Ran into trouble without this line, make sure to create the output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    summary_file = output_folder / "trained_summary.csv"
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Intitialize counters for statistics
    total_images = 0
    images_with_detections = 0
    total_detections = 0

    # Read the files and get statistics
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "detections"])

        for img_path in sorted(test_folder.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue

            total_images += 1
            results = model(img_path, imgsz=640, conf=0.10)
            boxes = results[0].boxes
            detections = 0 if boxes is None else len(boxes)

            if detections > 0:
                images_with_detections += 1
                total_detections += detections

            writer.writerow([img_path.name, detections])
            results[0].save(filename=str(output_folder / img_path.name))

    # Print results
    print("Test Metrics")
    print("Precision:", metrics.results_dict.get("metrics/precision(B)"))
    print("Recall:", metrics.results_dict.get("metrics/recall(B)"))
    print("mAP@50:", metrics.results_dict.get("metrics/mAP50(B)"))
    print("mAP@50-95:", metrics.results_dict.get("metrics/mAP50-95(B)"))

    # Print statistics for debugging and analysis
    print("Trained Model Summary")
    print("Images processed:", total_images)
    print("Images with detections:", images_with_detections)
    print("Images without detections:", total_images - images_with_detections)
    print("Total predicted boxes:", total_detections)
    print("CSV saved to:", summary_file)

