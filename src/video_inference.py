import argparse
from pathlib import Path
import sys
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_768_best.pt"
if not DEFAULT_MODEL_PATH.exists():
    FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_rdd2020_768_best.pt"
    if FALLBACK_MODEL_PATH.exists():
        DEFAULT_MODEL_PATH = FALLBACK_MODEL_PATH

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions" / "video"

CLASS_NAMES = {
    0: "D00: Longitudinal Crack",
    1: "D10: Transverse Crack",
    2: "D20: Alligator Crack",
    3: "D40: Pothole",
}


def run_video_inference(video_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH, output_dir: str | Path = DEFAULT_OUTPUT_DIR, conf: float = 0.25, imgsz: int = 768):
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not video_path.exists() or not video_path.is_file():
        print(f"Error: Video file not found at '{video_path}'")
        return None

    if not model_path.exists():
        print(f"Error: Model weights not found at '{model_path}'")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"annotated_{video_path.name}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total frames: {total_frames if total_frames > 0 else 'Unknown'}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
            )

            annotated_frame = frame.copy()
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = CLASS_NAMES.get(class_id, model.names.get(class_id, f"Class {class_id}"))

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        label = f"{class_name} {confidence:.2f}"
                        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(
                            annotated_frame,
                            (x1, max(0, y1 - text_h - 4)),
                            (x1 + text_w, y1),
                            (0, 165, 255),
                            -1,
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

            out.write(annotated_frame)

            if total_frames > 0 and frame_idx % 20 == 0:
                print(f"Progress: {frame_idx}/{total_frames} frames ({frame_idx / total_frames:.1%})")
            elif total_frames <= 0 and frame_idx % 20 == 0:
                print(f"Processed {frame_idx} frames...")

    finally:
        cap.release()
        out.release()

    print(f"Finished processing. Total frames: {frame_idx}")
    print(f"Annotated video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run YOLO road damage inference on video.")
    parser.add_argument("video_path", nargs="?", default="", help="Path to input video file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to model weights")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Path to output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=768, help="Inference image size")

    args = parser.parse_args()

    video_input = args.video_path
    if not video_input:
        video_input = input("Enter video path: ").strip()

    if not video_input:
        print("Error: No video path provided.")
        sys.exit(1)

    result = run_video_inference(
        video_path=video_input,
        model_path=args.model,
        output_dir=args.output_dir,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
