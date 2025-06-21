"""
Configuration file for semantic segmentation training.
Contains all hyperparameters and settings for the training pipeline.
"""
import os
import torch

class Config:
    """Training configuration class"""
    
    # Dataset paths (using os.path.realpath for cross-platform compatibility)
    DATASET_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    TRAIN_IMAGE_DIR = os.path.realpath(os.path.join(DATASET_ROOT, 'train', 'images'))
    TRAIN_MASK_DIR = os.path.realpath(os.path.join(DATASET_ROOT, 'train', 'masks'))
    TEST_IMAGE_DIR = os.path.realpath(os.path.join(DATASET_ROOT, 'test', 'images'))
    TEST_MASK_DIR = os.path.realpath(os.path.join(DATASET_ROOT, 'test', 'masks'))
    
    # Model configuration
    MODEL_NAME = 'lraspp_mobilenet_v3_large'
    NUM_CLASSES = 2  # background (0) and card (1)
    INPUT_HEIGHT = 480
    INPUT_WIDTH = 640
    PRETRAINED = False  # Enable pretrained weights for better convergence
    
    # Training configuration
    BATCH_SIZE = 16  # Adjust based on GPU memory
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Mixed precision training (FP16)
    USE_AMP = True
    
    # Loss function weights
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.5
    
    # Optimizer and scheduler
    OPTIMIZER = 'adamw'
    SCHEDULER = 'cosine'
    WARMUP_EPOCHS = 5
    
    # Data augmentation
    USE_AUGMENTATION = True
    ROTATION_LIMIT = 15
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    SATURATION_LIMIT = 0.2
    HUE_LIMIT = 0.1
    
    # Training settings
    PATIENCE = 15  # Early stopping patience
    SAVE_EVERY = 10  # Save checkpoint every N epochs
    VALIDATE_EVERY = 1  # Validate every N epochs
    
    # Paths
    CHECKPOINT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), 'checkpoints'))
    LOG_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), 'logs'))
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Evaluation metrics
    METRICS = ['iou', 'dice', 'pixel_accuracy']
    
    # Model pruning configuration
    PRUNING_AMOUNT = 0.3  # 30% of parameters to prune
    PRUNING_STRUCTURED = False  # Use unstructured pruning by default
    PRUNING_FINE_TUNE_EPOCHS = 20
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"Dataset Root: {cls.DATASET_ROOT}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Input Size: {cls.INPUT_HEIGHT}x{cls.INPUT_WIDTH}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.USE_AMP}")
        print(f"Augmentation: {cls.USE_AUGMENTATION}")
        print("=" * 50)
