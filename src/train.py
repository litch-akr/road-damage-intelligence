from pathlib import Path
import torch
from ultralytics import YOLO

torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using ROCm GPU acceleration: {gpu_name} (device={device})")
    else:
        print("GPU not detected by PyTorch, falling back to CPU.")

    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=768,
        batch=-1,          # Optimized for 8GB VRAM (Navi 33) to maximize GPU compute occupancy
        device=device,     # Explicit ROCm GPU device target (device=0)
        workers=8,         # Parallel DataLoader workers for high I/O throughput
        amp=True,          # FP16 Automatic Mixed Precision (leveraging RDNA3 matrix accelerators)
        cache=False,       # Disabled RAM cache (dataset requires ~29GB RAM vs 16GB system RAM; DataLoader workers handle streaming)
        patience=10,
        project=str(OUTPUT_DIR),
        name="yolo26n_img768",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()