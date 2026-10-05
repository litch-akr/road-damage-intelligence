# Road Damage Intelligence

> End-to-end road damage detection and analysis system built with deep learning on the **RDD2020** benchmark.

---

## Overview

Road Damage Intelligence is an end-to-end computer vision and deep learning project dedicated to automated road surface distress detection. By leveraging modern YOLO object detection architectures, this system detects, localizes, and classifies multiple types of road damage (such as longitudinal cracks, transverse cracks, alligator cracking, and potholes) to assist municipal road maintenance and infrastructure inspection.

---

## Objectives

- **Automated Damage Detection**: Detect and classify road distress from single images and continuous video streams.
- **Controlled Benchmarking**: Systematically evaluate model variants across architectures (YOLO11n vs. YOLO26n) and resolutions (640px vs. 768px).
- **In-Depth Error Diagnostics**: Analyze false positives, false negatives, class confusion, and challenging edge cases.
- **Production Readiness**: Provide end-to-end inference scripts, reproducible training pipelines, and interactive deployment prototypes.

---

## Damage Categories (RDD2020)

| Class ID | Code | Category | Description |
| :---: | :---: | :--- | :--- |
| **0** | **D00** | Longitudinal Crack | Cracks running parallel to the direction of travel (wheelpaths or construction joints). |
| **1** | **D10** | Transverse Crack | Cracks running perpendicular to the roadway centerline, often thermal or reflective. |
| **2** | **D20** | Alligator Crack | Interconnected polygonal cracking patterns indicating base/structural fatigue. |
| **3** | **D40** | Pothole / Rutting / Bump | Localized bowl-shaped structural depressions and severe road surface hazards. |

---

## Benchmark Results

### Résultat principal

| Expérience | Modèle | Résolution | mAP@50 | mAP@50-95 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | YOLO11n | 640 | 0.5100 | 0.2290 | 0.5380 | 0.5030 |
| **Exp. 2** | YOLO26n | 640 | 0.5243 | 0.2409 | 0.5747 | 0.4972 |
| **Exp. 3** | YOLO26n | 768 | **0.5390** | **0.2450** | 0.5610 | **0.5180** |

#### Key Insights
- **Architecture Improvement**: Moving from baseline YOLO11n to YOLO26n at 640px yields an increase of **+1.43% mAP@50** and **+1.19% mAP@50-95** with substantially higher precision (0.5747).
- **Resolution Scaling**: Scaling input resolution from 640px to 768px on YOLO26n further improves detection performance to **0.5390 mAP@50** and **0.5180 Recall**, proving particularly effective for fine-grained crack detection and subtle distress patterns.

---

## Notebooks

- **RDD2020 Training & Diagnostics Notebook**: [`notebooks/road_damage_intelligence_rdd2020.ipynb`](notebooks/road_damage_intelligence_rdd2020.ipynb)  
  Complete, end-to-end notebook designed for GPU execution (e.g. Kaggle / Colab). Includes dataset discovery, ground truth auditing, YOLO26n training at 768px, training dynamics & convergence curves, quantitative evaluation, side-by-side Ground Truth vs. Prediction overlays, false positive/negative error analysis, and comparative prototype inference against YOLO11n.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
Convert RDD2020 Pascal VOC annotations to YOLO format:
```bash
python src/convert_rdd2020.py
```

### 3. Model Training
```bash
python src/train.py
```

### 4. Inference
Run inference on a single test image:
```bash
python src/inference.py
```

---

## Roadmap

- [x] Dataset audit and VOC-to-YOLO conversion pipeline
- [x] Baseline detection model (YOLO11n @ 640px)
- [x] Controlled experiments (YOLO26n @ 640px & 768px)
- [x] Detailed error analysis (False Positives / False Negatives)
- [x] Single image inference pipeline
- [ ] Video stream inference
- [ ] FastAPI backend service
- [ ] Interactive Web UI / Dashboard
- [ ] Docker containerization