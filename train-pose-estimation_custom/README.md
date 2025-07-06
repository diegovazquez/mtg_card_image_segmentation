# Corner Detection Training

This directory contains the training pipeline for the corner detection model using Lite-HRNet architecture.

## Overview

The training system is designed to detect four corners of Magic: The Gathering cards in images. The model uses a Lite-HRNet backbone for efficient and accurate corner detection.

## Features

- **Lite-HRNet Model**: Based on the efficient Lite-HRNet architecture
- **Mixed Precision Training**: Uses autocast with float16 precision for faster training
- **CUDA GPU Acceleration**: Optimized for GPU training
- **Advanced Augmentations**: Comprehensive data augmentation using Albumentations
- **Custom Metrics**: Corner-specific evaluation metrics
- **Early Stopping**: Prevents overfitting with learning rate reduction
- **Model Export**: Supports TorchScript and ONNX export for deployment

## File Structure

```
train/
├── README.md              # This documentation
├── train.py              # Main training script
├── model.py              # Lite-HRNet model definition
├── dataset.py            # Dataset class with augmentations
└── metrics.py            # Custom metrics and loss functions
```

## Model Architecture

- **Input**: 480 (width) × 640 (height) RGB images
- **Backbone**: Lite-HRNet (efficient variant of HRNet)
- **Output**: 8 coordinates representing 4 corners [x1,y1,x2,y2,x3,y3,x4,y4]
- **Training**: Mixed precision with autocast for efficiency

## Dataset Requirements

The training expects the following dataset structure:

```
dataset/
├── corner_annotations.json    # Corner coordinates for all images
├── train/
│   └── images/
│       ├── card_image_001.jpg
│       ├── card_image_002.jpg
│       └── ...
└── test/
    └── images/
        ├── card_image_001.jpg
        ├── card_image_002.jpg
        └── ...
```

### Annotations Format

The `corner_annotations.json` file contains corner coordinates:

```json
{
  "train": {
    "image_name.jpg": [
      [x1, y1],  // Top-left corner
      [x2, y2],  // Top-right corner  
      [x3, y3],  // Bottom-right corner
      [x4, y4]   // Bottom-left corner
    ]
  },
  "test": {
    "image_name.jpg": [
      [x1, y1], [x2, y2], [x3, y3], [x4, y4]
    ]
  }
}
```

## Data Augmentations

The training uses comprehensive augmentations via Albumentations:

- **Geometric**: Horizontal flip, affine transforms, elastic deformation, grid distortion
- **Color**: Color jitter, brightness/contrast adjustment
- **Noise**: Gaussian noise and blur
- **Probability-based**: Each augmentation has configurable probability

## Training Configuration

Default configuration:

```python
{
    'num_epochs': 200,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'lr_factor': 0.5,
    'lr_patience': 10,
    'early_stopping_patience': 20,
    'mse_weight': 1.0,
    'penalty_weight': 0.1,
    'image_size': (480, 640),
    'num_workers': 4,
    'output_dir': './outputs',
    'dataset_path': '../dataset'
}
```

## Usage

### Basic Training

```bash
python train.py --dataset /path/to/dataset --output /path/to/outputs
```

### Advanced Training

```bash
python train.py \\
    --dataset /path/to/dataset \\
    --output /path/to/outputs \\
    --epochs 300 \\
    --batch-size 64 \\
    --lr 0.001 \\
    --device cuda
```

### With Custom Configuration

```bash
python train.py --config config.json --dataset /path/to/dataset
```

Example `config.json`:

```json
{
    "num_epochs": 300,
    "batch_size": 64,
    "learning_rate": 0.0005,
    "weight_decay": 1e-4,
    "image_size": [480, 640],
    "early_stopping_patience": 30
}
```

## Metrics

The training tracks several corner-specific metrics:

- **corner_acc_3px**: Percentage of predictions within 3 pixels of ground truth
- **corner_acc_6px**: Percentage of predictions within 6 pixels of ground truth  
- **mean_corner_distance**: Average pixel distance between predicted and true corners

## Loss Function

Custom loss function combining:

- **MSE Loss**: Basic coordinate regression loss
- **Boundary Penalty**: Penalizes predictions outside valid coordinate range
- **Distance Penalty**: Penalizes unrealistic corner arrangements

## Training Features

### Mixed Precision Training

Uses PyTorch's autocast for automatic mixed precision:

```python
with autocast():
    predictions = model(images)
    loss = criterion(predictions, targets)
```

### Early Stopping

Monitors validation loss with configurable patience:

- Stops training if no improvement for specified epochs
- Restores best model weights automatically

### Learning Rate Scheduling

Uses ReduceLROnPlateau scheduler:

- Reduces learning rate when validation loss plateaus
- Configurable reduction factor and patience

### Model Checkpointing

Saves checkpoints at multiple points:

- `checkpoint_epoch_N.pth`: Regular epoch checkpoints
- `best_model.pth`: Best model based on validation loss
- `final_model.pth`: Final model after training completion

## Output Files

After training, the following files are generated:

- `best_model.pth`: Best model checkpoint
- `final_model.pth`: Final model checkpoint
- `training_history.json`: Training metrics history
- `model_traced.pt`: TorchScript exported model
- `model.onnx`: ONNX exported model

## Model Export

The training script automatically exports models in multiple formats:

### TorchScript Export

```python
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('model_traced.pt')
```

### ONNX Export

```python
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    opset_version=11,
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

## Platform Compatibility

The training scripts use `os.path.realpath()` for path resolution, ensuring compatibility across different platforms and Python interpreters.

## Requirements

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- timm >= 0.6.0
- albumentations >= 1.0.0
- opencv-python >= 4.5.0
- numpy >= 1.20.0
- tqdm >= 4.60.0
- Pillow >= 8.0.0

## GPU Requirements

- CUDA-capable GPU with at least 8GB VRAM recommended
- Mixed precision training reduces memory usage significantly
- Batch size can be adjusted based on available GPU memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Dataset Loading Error**: Verify dataset structure and annotation format
3. **Model Export Failure**: Check PyTorch version compatibility

### Performance Tips

- Use SSD storage for dataset to reduce I/O bottleneck
- Increase num_workers for faster data loading
- Use larger batch sizes on high-memory GPUs
- Monitor GPU utilization during training

## Monitoring Training

Training progress is displayed with:

- Real-time progress bars using tqdm
- Epoch-wise metrics logging
- Learning rate scheduling information
- Early stopping notifications

## Example Training Session

```bash
$ python train.py --dataset ../dataset --output ./outputs --epochs 200

Using device: cuda
Creating data loaders...
Train samples: 5000
Val samples: 1000
Creating model...
Model parameters: 1,234,567

Starting training for 200 epochs...
Epoch 1/200: 100%|████████| 157/157 [02:34<00:00, 1.02it/s, loss=0.0234, lr=0.001000]
Validation: 100%|████████| 32/32 [00:15<00:00, 2.11it/s]

Epoch 1/200:
Train Loss: 0.0234
Val Loss: 0.0198
Train Metrics: {'corner_acc_3px': 45.2, 'corner_acc_6px': 67.8, 'mean_corner_distance': 4.12}
Val Metrics: {'corner_acc_3px': 48.1, 'corner_acc_6px': 69.3, 'mean_corner_distance': 3.89}
```

This comprehensive training pipeline provides everything needed to train a robust corner detection model for Magic: The Gathering cards.