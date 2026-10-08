import io
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_768_best.pt"
if not DEFAULT_MODEL_PATH.exists():
    FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_rdd2020_768_best.pt"
    if FALLBACK_MODEL_PATH.exists():
        DEFAULT_MODEL_PATH = FALLBACK_MODEL_PATH

CONF_THRESHOLD = 0.25
IMGSZ = 768

app = FastAPI(
    title="Road Damage Intelligence API",
    description="Inference API for road surface damage detection using YOLO.",
    version="1.0.0",
)

model: YOLO | None = None


@app.on_event("startup")
def load_model():
    global model
    if not DEFAULT_MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at '{DEFAULT_MODEL_PATH}'")
    model = YOLO(str(DEFAULT_MODEL_PATH))


@app.get("/health")
def health_check() -> Dict[str, str]:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return {"status": "ok", "model": DEFAULT_MODEL_PATH.name}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded",
            )
        image = Image.open(io.BytesIO(contents))
        image.verify()
        # Re-open for actual processing because verify() changes internal state
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image",
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading image: {str(e)}",
        )

    results = model.predict(
        source=image,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names.get(class_id, f"Class {class_id}")
                x1, y1, x2, y2 = [round(coord, 2) for coord in box.xyxy[0].tolist()]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )

    return {
        "detection_count": len(detections),
        "detections": detections,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
