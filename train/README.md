# MTG Card Semantic Segmentation Training

This directory contains a complete training pipeline for semantic segmentation of Magic: The Gathering cards using PyTorch and the LR-ASPP (Low-Resolution Atrous Spatial Pyramid Pooling) architecture with MobileNetV3-Large backbone.

## Overview

The training system implements:
- **FP16 Mixed Precision Training** for memory efficiency and speed
- **LR-ASPP Architecture** optimized for mobile deployment
- **Comprehensive Data Augmentation** for robust generalization
- **Model Pruning** for deployment optimization
- **Multi-format Export** (ONNX, TorchScript, PyTorch)
- **Advanced Evaluation** with detailed metrics and visualization

## Architecture

- **Model**: LR-ASPP with MobileNetV3-Large backbone
- **Input Resolution**: 480x640 (vertical orientation, perfect for MTG cards)
- **Classes**: 2 (background=0, card=1)
- **Precision**: FP16 mixed precision training
- **Loss**: Combined Dice + Cross-entropy loss

## Directory Structure

```
train/
├── README.md                 # This documentation
├── config.py                 # Training configuration
├── train.py                  # Main training script
├── model.py                  # LR-ASPP model implementation
├── dataset.py                # Dataset loader with augmentations
├── utils.py                  # Training utilities and metrics
├── prune.py                  # Model pruning script
├── evaluate.py               # Model evaluation script
├── export.py                 # Model export utilities
├── checkpoints/              # Model checkpoints (auto-created)
├── logs/                     # Training logs (auto-created)
└── exported_models/          # Exported models (auto-created)
```

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate python virtual enviroment
python3.10 -m venv venv_ml
source venv_ml/bin/activate

# Upgrade pip package manager 
python -m pip install --upgrade pip

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install albumentations opencv-python pillow numpy
pip install matplotlib seaborn scikit-learn
pip install tqdm wandb

# Optional (for export functionality)
pip install onnx onnxruntime onnxoptimizer

# Create dirs
mkdir train/checkpoints
mkdir train/logs

```

### 2. Prepare Dataset

Ensure your dataset follows this structure relative to the `train/` directory:

```
../dataset/
├── train/
│   ├── image/          # Training images (.jpg)
│   └── mask/           # Training masks (.png)
└── test/
    ├── image/          # Test images (.jpg)
    └── mask/           # Test masks (.png)
```

**Important**: 
- Images should be in JPG format
- Masks should be in PNG format with white pixels (255) for cards and black pixels (0) for background
- Image and mask filenames should match (except for extensions)

### 3. Start Training

```bash
# Basic training
cd train
python train.py

# Training with Weights & Biases logging
python train.py --use-wandb

# Resume from checkpoint
python train.py --resume checkpoints/best_model.pth
```

## Configuration

All training parameters are centralized in `config.py`. Key settings include:

```python
# Model configuration
MODEL_NAME = 'lraspp_mobilenet_v3_large'
INPUT_HEIGHT = 480
INPUT_WIDTH = 640
NUM_CLASSES = 2

# Training configuration
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
USE_AMP = True  # FP16 mixed precision

# Data augmentation
USE_AUGMENTATION = True
ROTATION_LIMIT = 15
BRIGHTNESS_LIMIT = 0.2
```

## Training Features

### Mixed Precision Training (FP16)

Automatic Mixed Precision (AMP) is enabled by default for:
- **Memory Efficiency**: ~50% reduction in GPU memory usage
- **Speed Improvement**: ~1.5-2x faster training on modern GPUs
- **Maintained Accuracy**: Gradient scaling prevents underflow

### Data Augmentation

Comprehensive augmentation pipeline includes:
- **Geometric**: Rotation, scaling, horizontal flip, elastic transforms
- **Photometric**: Brightness, contrast, saturation, hue adjustments
- **Noise**: Gaussian noise and blur for robustness
- **Proper Mask Handling**: All augmentations preserve mask consistency

### Advanced Training Features

- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Gradient Scaling**: Automatic handling for FP16 training
- **Model Checkpointing**: Best model and periodic saves
- **Comprehensive Logging**: Training curves and detailed metrics

## Model Pruning

Reduce model size for deployment while maintaining accuracy:

```bash
# Unstructured pruning (30% sparsity)
python prune.py --model-path checkpoints/best_model.pth --pruning-amount 0.3

# Structured pruning with fine-tuning
python prune.py --model-path checkpoints/best_model.pth --structured --fine-tune

# Advanced pruning options
python prune.py --model-path checkpoints/best_model.pth \
                --structured \
                --pruning-amount 0.5 \
                --fine-tune \
                --fine-tune-epochs 30 \
                --remove-masks
```

### Pruning Options

- **Unstructured Pruning**: Zero out individual weights (better compression)
- **Structured Pruning**: Remove entire channels/filters (better hardware acceleration)
- **Fine-tuning**: Recover performance after pruning
- **Mask Removal**: Permanent pruning for deployment

## Model Evaluation

Comprehensive evaluation with detailed metrics and visualizations:

```bash
# Basic evaluation
python evaluate.py --model-path checkpoints/best_model.pth

# Full analysis with visualizations
python evaluate.py --model-path checkpoints/best_model.pth \
                   --analyze-predictions \
                   --find-failures \
                   --save-plots
```

### Evaluation Features

- **Per-class Metrics**: Precision, Recall, F1-score, IoU for each class
- **Confusion Matrix**: Visual representation of classification performance
- **Prediction Visualization**: Side-by-side comparison of predictions vs ground truth
- **Failure Case Analysis**: Identify and analyze worst-performing samples
- **Confidence Maps**: Visualize model uncertainty

## Model Export

Export trained models for deployment:

```bash
# Create complete deployment package
python export.py --model-path checkpoints/best_model.pth --create-package

# Export specific formats
python export.py --model-path checkpoints/best_model.pth \
                 --export-onnx \
                 --export-torchscript \
                 --dynamic-batch
```

### Export Formats

1. **ONNX**: Cross-platform deployment with optimization
2. **TorchScript**: PyTorch-native deployment format
3. **State Dict**: PyTorch weights for further training/fine-tuning

### Deployment Package Contents

- `card_segmentation.onnx`: Optimized ONNX model
- `card_segmentation.pt`: TorchScript model
- `card_segmentation_state_dict.pth`: PyTorch state dict
- `inference_example.py`: Ready-to-use inference script
- `README.md`: Deployment documentation

## Usage Examples

### Training from Scratch

```bash
cd train
python train.py --use-wandb
```

### Resume Training

```bash
python train.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Evaluate Model

```bash
python evaluate.py --model-path checkpoints/best_model.pth --analyze-predictions
```

### Prune and Fine-tune

```bash
python prune.py --model-path checkpoints/best_model.pth \
                --pruning-amount 0.3 \
                --fine-tune \
                --fine-tune-epochs 20
```

### Export for Deployment

```bash
python export.py --model-path checkpoints/best_model.pth --create-package
```

## Performance Expectations

### Model Size and Speed

- **Original Model**: ~3.2M parameters, ~12.8 MB
- **After 30% Pruning**: ~2.2M parameters, ~8.9 MB
- **Inference Speed**: ~10-15 FPS on mobile devices
- **Training Time**: ~2-4 hours on modern GPU (depends on dataset size)

### Accuracy Targets

- **IoU (Card Class)**: >0.85 on test set
- **Pixel Accuracy**: >0.95 overall
- **Dice Coefficient**: >0.90 for card segmentation

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `BATCH_SIZE` in `config.py`
   - Enable mixed precision: `USE_AMP = True`
   - Reduce `NUM_WORKERS` for data loading

2. **Slow Training**:
   - Ensure CUDA is available: Check `Config.DEVICE`
   - Use mixed precision: `USE_AMP = True`
   - Optimize data loading: Increase `NUM_WORKERS`

3. **Poor Convergence**:
   - Check dataset quality and mask consistency
   - Adjust learning rate: `LEARNING_RATE`
   - Increase training epochs: `NUM_EPOCHS`

4. **Dataset Loading Errors**:
   - Verify dataset paths in `config.py`
   - Check image/mask filename matching
   - Ensure proper file formats (JPG/PNG)

### Memory Optimization

```python
# In config.py, reduce memory usage:
BATCH_SIZE = 4          # Reduce batch size
NUM_WORKERS = 2         # Reduce data loading workers
USE_AMP = True          # Enable mixed precision
PIN_MEMORY = False      # Disable if low on system RAM
```

### Performance Monitoring

- Use `--use-wandb` for online experiment tracking
- Monitor `train/logs/` for local training history
- Check GPU utilization with `nvidia-smi`

## Cross-Platform Compatibility

All scripts use `os.path.realpath()` for cross-platform path handling:
- **Windows**: Full compatibility with Windows paths
- **Linux/macOS**: Native Unix path support

## Advanced Features

### Custom Loss Functions

The training system uses a combined loss function:
```python
loss = dice_weight * dice_loss + ce_weight * cross_entropy_loss
```

Weights can be adjusted in `config.py`:
```python
DICE_WEIGHT = 0.5  # Dice loss weight
BCE_WEIGHT = 0.5   # Cross-entropy weight
```

### Learning Rate Scheduling

Multiple scheduling options available:
- **Cosine Annealing**: Smooth decay with restarts
- **Step Decay**: Periodic learning rate reduction
- **Warmup**: Gradual learning rate increase at start

### Experiment Tracking

Integration with Weights & Biases:
```bash
python train.py --use-wandb
```

Tracks:
- Training/validation losses
- Metrics (IoU, Dice, Accuracy)
- Learning rate schedules
- Model artifacts

## Support

For issues and questions:
1. Check this documentation first
2. Review the troubleshooting section
3. Open an issue in the main repository
4. Provide detailed error messages and system information
