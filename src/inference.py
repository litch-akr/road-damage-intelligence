from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_rdd2020_768_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"


def main():
    model = YOLO(str(MODEL_PATH))

    image_path = input("Enter image path: ").strip()

    results = model.predict(
        source=image_path,
        imgsz=768,
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="inference",
        exist_ok=True,
    )
    for result in results:
        print("\nDetections:")

        if result.boxes is None or len(result.boxes) == 0:
            print("No road damage detected.")
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"- {class_name}: "
                f"{confidence:.2f} "
                f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            )

    print(f"\nAnnotated image saved to:")
    print(OUTPUT_DIR / "inference")


if __name__ == "__main__":
    main()