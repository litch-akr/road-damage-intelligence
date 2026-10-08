# Road Damage Intelligence

> Automated road surface distress detection, comparative deep learning benchmarking, and production inference system built on the **RDD2020** benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset & Distress Taxonomy](#dataset--distress-taxonomy)
- [Model Architecture & Experiments](#model-architecture--experiments)
- [Benchmark Results](#benchmark-results)
- [Error Analysis & Diagnostics](#error-analysis--diagnostics)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Data Preparation](#data-preparation)
- [Training Pipeline](#training-pipeline)
- [Inference & Deployment](#inference--deployment)
  - [Single Image Inference](#single-image-inference)
  - [Video Stream Inference](#video-stream-inference)
  - [Interactive Gradio Prototype](#interactive-gradio-prototype)
  - [FastAPI Inference Server](#fastapi-inference-server)
- [Research Notebooks](#research-notebooks)
- [License](#license)

---

## Overview

Road surface deterioration compromises vehicular safety, inflates transportation operating costs, and accelerates civil infrastructure degradation. **Road Damage Intelligence** is a production-oriented computer vision system engineered for automated detection, classification, and localization of pavement distress.

Leveraging modern YOLO architectures (YOLO11n and YOLO26n) trained on the multi-national **RDD2020** dataset, this repository provides:
1. End-to-end data conversion from Pascal VOC XML annotations to YOLO format.
2. Controlled experimental benchmarking across model architectures and input resolutions (640px vs. 768px).
3. In-depth quantitative and qualitative error diagnostics (confusion matrices, PR curves, false positives/negatives analysis).
4. Dual deployment modalities: a RESTful FastAPI backend and an interactive Gradio web interface.
5. High-throughput frame-by-frame video inference preserving native frame rates and resolutions.

---

## Key Features

- **Multi-Country Road Generalization**: Evaluated on diverse pavement textures, lighting variations, and roadway conditions across international RDD2020 subsets (Czech Republic, India, Japan).
- **Resolution Scaling & Precision Optimization**: High-resolution input processing (768x768) to capture fine-grained longitudinal and transverse hairline fractures.
- **Robust Inference Pipelines**: Python CLI tools for batch image and continuous video processing.
- **Standardized REST API**: Production-ready FastAPI microservice with input validation, health probing, and structured bounding box payloads.
- **Self-Contained Training & Diagnostic Notebooks**: Interactive Jupyter workflow with dataset auditing, convergence monitoring, and Ground Truth vs. Prediction side-by-side overlays.

---

## Dataset & Distress Taxonomy

The project utilizes the **RDD2020 (Road Damage Dataset 2020)** benchmark, which categorizes pavement distress into four primary classes:

| Class ID | Code | Category | Morphological Description | Severity & Impact |
| :---: | :---: | :--- | :--- | :--- |
| **0** | **D00** | Longitudinal Crack | Cracks parallel to roadway direction (wheelpaths, construction joints). | Water ingress risk; early distress indicator. |
| **1** | **D10** | Transverse Crack | Cracks perpendicular to roadway centerline (thermal or reflective). | Joint deterioration; structural subgrade weakness. |
| **2** | **D20** | Alligator Crack | Interconnected polygonal cracking patterns. | Structural fatigue failure of base course. |
| **3** | **D40** | Pothole / Rutting / Bump | Localized bowl-shaped structural depressions and surface loss. | Immediate vehicular safety hazard. |

### Data Split Specification
- **Split Ratio**: 80% Training / 20% Validation (stratified by country subset).
- **Random Seed**: Fixed at `42` for strict experimental reproducibility.
- **Format**: Converted from Pascal VOC XML to normalized YOLO format (`[class_id, x_center, y_center, width, height]`).

---

## Model Architecture & Experiments

We evaluate lightweight, real-time object detection models suitable for edge and server-side deployment:

1. **YOLO11n (Baseline)**: Compact baseline model evaluated at standard resolution ($640 \times 640$).
2. **YOLO26n @ 640px**: Next-generation anchor-free compact detector evaluated under identical training hyperparameters.
3. **YOLO26n @ 768px (Champion)**: High-resolution variant configured to enhance feature map granularity for fine crack detection.

---

## Benchmark Results

All models were evaluated on the RDD2020 validation set under uniform confidence and IoU thresholds.

### Quantitative Summary

| Experiment | Model Architecture | Resolution | mAP@50 | mAP@50-95 | Precision | Recall | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | YOLO11n | 640 | 0.5100 | 0.2290 | 0.5380 | 0.5030 | ~2.6M |
| **Exp. 2** | YOLO26n | 640 | 0.5243 | 0.2409 | **0.5747** | 0.4972 | ~2.4M |
| **Exp. 3 (Final)** | YOLO26n | **768** | **0.5390** | **0.2450** | 0.5610 | **0.5180** | ~2.4M |

### Performance Analysis

- **Architecture Efficacy**: Transitioning from YOLO11n to YOLO26n at $640 \times 640$ delivered a **+1.43% improvement in mAP@50** and **+1.19% in mAP@50-95**, accompanied by a substantial gain in Precision (+3.67%).
- **Resolution Scaling**: Increasing input resolution from 640px to 768px on YOLO26n further boosted detection performance to **0.5390 mAP@50** and **0.5180 Recall** (+2.08% Recall gain over 640px), resolving fine crack textures that were previously missed.

---

## Error Analysis & Diagnostics

Our diagnostic evaluation identified key failure modes and operational boundaries:

1. **Class Confusion (D00 vs. D10)**: Minor confusion between longitudinal and transverse cracks occurs primarily at curved pavement sections and diagonal fissures where dominant orientation is ambiguous.
2. **Fine-Grained Alligator Cracks (D20)**: Complex multi-directional cracks occasionally suffer from fragmented bounding box predictions.
3. **Pothole Detection (D40)**: Potholes exhibit the highest detection certainty owing to distinct morphological shadows, depth boundaries, and shape distinctiveness.
4. **Environmental Edge Cases**: Wet pavement specular reflections, shadowed tree canopies, and faded lane markings constitute the primary sources of false positive activations.

---

## Project Structure

```
road-damage-intelligence/
├── configs/                   # Configuration definitions
├── data/                      # Dataset directory
│   ├── train/                 # Raw Pascal VOC subsets (Czech, India, Japan)
│   └── yolo/                  # Converted YOLO dataset (train/val splits & data.yaml)
├── models/                    # Exported model weights (.pt checkpoints)
│   ├── yolo26n_rdd2020_768_best.pt   # Champion model (YOLO26n @ 768px)
│   ├── yolo26n_rdd2020_640_best.pt   # Comparison model (YOLO26n @ 640px)
│   └── rdd2020_yolo_best.pt          # Baseline weights
├── notebooks/                 # Research and diagnostic notebooks
│   └── road_damage_intelligence_rdd2020.ipynb
├── outputs/                   # Training runs, validation plots, predictions
├── src/                       # Core source code
│   ├── __init__.py
│   ├── api.py                 # FastAPI REST backend service
│   ├── convert_rdd2020.py     # VOC XML to YOLO converter with split generator
│   ├── inference.py           # Single-image inference CLI
│   ├── prototype.py           # Gradio interactive web prototype
│   ├── train.py               # Local training runner
│   └── video_inference.py     # Frame-by-frame video detection engine
├── tests/                     # Verification and test suites
│   └── test_verification.py   # API and inference integration tests
├── requirements.txt           # Python runtime dependencies
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA or ROCm recommended for hardware acceleration)

### Environment Setup

```bash
git clone https://github.com/akramallak/road-damage-intelligence.git
cd road-damage-intelligence

python -m venv venv
# On Linux / macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Preparation

To convert the raw RDD2020 dataset from Pascal VOC format into the required YOLO format:

```bash
python src/convert_rdd2020.py
```

This script:
- Parses XML annotations across `Czech`, `India`, and `Japan` directories.
- Converts coordinates to normalized YOLO bounding boxes.
- Performs a reproducible 80/20 train/validation split (`seed=42`).
- Generates `data/yolo/data.yaml` pointing to image and label paths.

---

## Training Pipeline

To initiate model training using the local runner:

```bash
python src/train.py
```

Training parameters are configured for optimal throughput:
- Architecture: YOLO26n (`yolo26n.pt`)
- Input Size: `768x768`
- Epochs: `50` (with early stopping patience of 10)
- Hardware acceleration: Auto-detects CUDA/ROCm with FP16 mixed precision (`amp=True`).

---

## Inference & Deployment

### Single Image Inference

Run detection on a single image via the CLI:

```bash
python src/inference.py
```
*Prompts for an image path and outputs detected bounding box coordinates, class labels, confidences, and saved visualizations.*

### Video Stream Inference

Process complete video files with frame-by-frame detection:

```bash
# Direct execution with argument
python src/video_inference.py path/to/input_video.mp4

# Custom model and confidence threshold
python src/video_inference.py path/to/input_video.mp4 --conf 0.30 --imgsz 768
```
*Annotated videos are automatically exported to `outputs/predictions/video/` preserving input resolution and FPS.*

### Interactive Gradio Prototype

Launch a browser-based user interface for live testing:

```bash
python src/prototype.py
```
*Opens a local web server (typically at `http://127.0.0.1:7860`) with image upload and bounding box rendering.*

### FastAPI Inference Server

Launch the production REST API microservice:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

#### Health Check
```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model": "yolo26n_rdd2020_768_best.pt"
}
```

#### Image Prediction Endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_road.jpg"
```

Response payload:
```json
{
  "detection_count": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "D00",
      "confidence": 0.8412,
      "box": {
        "x1": 142.5,
        "y1": 210.3,
        "x2": 315.8,
        "y2": 480.1
      }
    },
    {
      "class_id": 3,
      "class_name": "D40",
      "confidence": 0.7935,
      "box": {
        "x1": 420.0,
        "y1": 350.2,
        "x2": 512.4,
        "y2": 440.6
      }
    }
  ]
}
```

---

## Research Notebooks

The full experimental workflow, from data exploration to comparative analytics, is documented in:
- **`notebooks/road_damage_intelligence_rdd2020.ipynb`**
  - Dataset discovery, annotation verification, class distribution inspection.
  - Model training dynamics, loss convergence, PR curves.
  - Side-by-side Ground Truth vs. Prediction visualization.
  - Quantitative error taxonomy and side-by-side YOLO11n vs. YOLO26n qualitative comparison.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.