import io
import tempfile
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app, load_model
from src.video_inference import run_video_inference


def test_fastapi_endpoints():
    print("Testing FastAPI app startup and endpoints...")
    # Trigger startup event
    load_model()

    client = TestClient(app)

    # 1. Health check
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    health_data = response.json()
    assert health_data.get("status") == "ok", f"Expected ok, got {health_data}"
    print(f"GET /health passed: {health_data}")

    # 2. Predict check with valid image
    sample_img_path = Path("data/train/Czech/images/Czech_000000.jpg")
    if sample_img_path.exists():
        with open(sample_img_path, "rb") as f:
            files = {"file": ("Czech_000000.jpg", f, "image/jpeg")}
            response = client.post("/predict", files=files)
    else:
        # Create a synthetic image if dataset image not found
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        files = {"file": ("test.jpg", buf, "image/jpeg")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    predict_data = response.json()
    assert "detection_count" in predict_data, "Missing detection_count"
    assert "detections" in predict_data, "Missing detections"
    assert isinstance(predict_data["detections"], list), "detections should be list"
    print(f"POST /predict passed: {predict_data}")

    # 3. Predict check with invalid image
    bad_files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/predict", files=bad_files)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("POST /predict with invalid file passed (400 Bad Request)")


def test_video_inference():
    print("\nTesting video inference...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video_path = Path(tmpdir) / "test_input.mp4"
        out_dir = Path(tmpdir) / "output"

        # Generate a small 10-frame test video
        height, width = 480, 640
        fps = 10.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video_path), fourcc, fps, (width, height))

        sample_img_path = Path("data/train/Czech/images/Czech_000000.jpg")
        if sample_img_path.exists():
            frame = cv2.imread(str(sample_img_path))
            frame = cv2.resize(frame, (width, height))
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        for _ in range(10):
            writer.write(frame)
        writer.release()

        # Run inference
        output_path = run_video_inference(
            video_path=tmp_video_path,
            output_dir=out_dir,
            conf=0.25,
            imgsz=768,
        )

        assert output_path is not None, "Output path was None"
        assert output_path.exists(), f"Output file does not exist: {output_path}"
        assert output_path.stat().st_size > 0, "Output file is empty"
        print(f"Video inference passed. Output saved: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    test_fastapi_endpoints()
    test_video_inference()
    print("\nAll verifications succeeded!")
