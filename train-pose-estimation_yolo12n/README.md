# Magic: The Gathering Card Corner Detection Training

This directory contains a complete training pipeline for detecting the four corners of Magic: The Gathering cards using YOLO11n-pose architecture. The system is optimized for web and mobile deployment with ONNX export capabilities.

## 🎯 Overview

The corner detection model identifies the four corners of MTG cards with high precision, enabling applications like:
- Card scanning and digitization
- Perspective correction
- Card recognition systems
- Automated card cataloging

### Key Features

- **YOLO11n-pose Architecture**: Lightweight pose estimation optimized for corner detection
- **Advanced Data Augmentation**: Progressive augmentation strategies for robust training
- **Early Stopping & LR Scheduling**: Prevent overfitting and optimize convergence
- **Distance-based Metrics**: Evaluation at 5px, 10px, and 20px accuracy thresholds
- **ONNX Export**: Web and mobile deployment ready
- **Comprehensive Evaluation**: Detailed performance analysis and visualization

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install ultralytics torch torchvision opencv-python
pip install albumentations matplotlib seaborn pandas
pip install onnx onnxruntime scipy tqdm pyyaml
```

### 1. Basic Training

```bash
# Train with default settings
python train.py \
    --annotations ../dataset/corner_annotations.json \
    --images-dir ../dataset \
    --epochs 200 \
    --batch-size 24 \
    --model yolo12n-pose.yaml
    --device cuda
```

### 2. Evaluate Model

```bash
# Evaluate trained model
python evaluate_model.py
    --model runs/train/corner_detection_20250627_170134/weights/best.pt  \
    --annotations ../dataset/corner_annotations.json   \
    --images-dir ../dataset  \
    --split test
```

### 3. Export to ONNX

```bash
yolo export format=onnx model=runs/train/corner_detection_20250630_234018/weights/best.pt device=0 dynamic=True half=True

yolo export format=onnx imgsz=480,640 model=runs/train/corner_detection_20250630_234018/weights/best.pt device=0 dynamic=True half=True
```

## 📊 Dataset Information

### Data Format

The training data consists of:
- **Training Images**: 8,800 images in `dataset/train/images/`
- **Test Images**: 2,180 images in `dataset/test/images/`
- **Annotations**: Corner coordinates in `dataset/corner_annotations.json`

### Annotation Format

```json
{
  "train": {
    "image_name.jpg": [
      [x1, y1],  // Top-left corner
      [x2, y2],  // Top-right corner
      [x3, y3],  // Bottom-right corner
      [x4, y4]   // Bottom-left corner
    ]
  }
}
```

### Image Specifications

- **Resolution**: 480×640 pixels (Portrait orientation)
- **Format**: JPG images
- **Card Types**: Full-art and normal Magic: The Gathering cards
- **Corner Order**: Top-left → Top-right → Bottom-right → Bottom-left

## 🧠 Model Architecture

### YOLO11n-pose Configuration

- **Input Size**: 640×640 pixels
- **Keypoints**: 4 corners with (x, y, visibility)
- **Classes**: 1 (card)
- **Parameters**: ~3M parameters
- **FLOPs**: ~8.7G

### Model Output

- **Bounding Box**: Card detection box
- **Keypoints**: 4 corner coordinates with confidence scores
- **Confidence**: Overall detection confidence

## 🔧 Training Configuration

### Default Settings

```python
# Model
model_name = "yolo11n-pose.pt"
img_size = 640
num_keypoints = 4

# Training
epochs = 200
batch_size = 16
learning_rate = 0.01
weight_decay = 0.0005

# Early Stopping
patience = 30
min_delta = 0.001

# Learning Rate Scheduling
lr_patience = 10
lr_factor = 0.5
```

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (4GB+ VRAM)
- **RAM**: 8GB+ system memory
- **Storage**: 10GB+ free space for datasets and outputs
- **Training Time**: ~2-4 hours on RTX 3080

## 📈 Evaluation Metrics

### Distance-based Accuracy

The model is evaluated using pixel distance thresholds:

- **5px Accuracy**: Percentage of corners within 5 pixels of ground truth
- **10px Accuracy**: Percentage of corners within 10 pixels of ground truth  
- **20px Accuracy**: Percentage of corners within 20 pixels of ground truth

### Performance Metrics

- **Detection Rate**: Percentage of successful card detections
- **Mean Distance Error**: Average pixel error across all corners
- **Per-corner Accuracy**: Individual accuracy for each corner type
- **Inference Speed**: Processing time per image

### Expected Performance

Target metrics for a well-trained model:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| 5px Accuracy | >80% | >85% | >90% |
| 10px Accuracy | >90% | >95% | >98% |
| 20px Accuracy | >95% | >98% | >99% |
| Detection Rate | >95% | >98% | >99% |
| Mean Error | <8px | <5px | <3px |

## 🌐 Deployment