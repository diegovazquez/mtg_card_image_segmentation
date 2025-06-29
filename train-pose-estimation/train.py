#!/usr/bin/env python3
"""
Training script for Magic: The Gathering card corner detection using YOLO11n-pose.
Implements early stopping, learning rate scheduling, and comprehensive logging.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from model import CornerDetectionModel, create_data_config
from dataset import create_yolo_annotations
from augmentation import CornerAugmentation
from evaluate_model import CornerEvaluator


class TrainingConfig:
    """Training configuration class."""
    
    def __init__(self):
        # Model configuration
        self.model_name = "yolo11n-pose.pt"
        self.num_keypoints = 4
        self.img_size = 640
        
        # Training configuration
        self.epochs = 200
        self.batch_size = 16
        self.learning_rate = 0.01
        self.weight_decay = 0.0005
        self.momentum = 0.937
        self.warmup_epochs = 3
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1
        
        # Early stopping configuration
        self.patience = 30
        self.min_delta = 0.001
        self.restore_best_weights = True
        
        # Learning rate scheduling
        self.lr_scheduler = "ReduceLROnPlateau"
        self.lr_patience = 10
        self.lr_factor = 0.5
        self.lr_min = 1e-6
        
        # Data configuration
        self.train_split = 0.8
        self.val_split = 0.2
        self.augmentation_mode = "progressive"  # 'light', 'medium', 'heavy', 'progressive'
        #self.augmentation_mode = "light"  # 'light', 'medium', 'heavy', 'progressive'

        # Hardware configuration
        self.device = "auto"
        self.workers = 8
        self.use_cuda = True
        
        # Logging and saving
        self.save_dir = "runs/train"
        self.save_period = 10
        self.log_level = "INFO"
        self.plot_results = True
        self.save_best_only = False
        
        # Advanced training options
        self.mixed_precision = True
        self.gradient_clipping = True
        self.clip_value = 10.0
        self.label_smoothing = 0.0
        
        # Validation configuration
        self.val_interval = 1
        self.val_metrics = ["precision", "recall", "mAP50", "mAP50-95", "corner_accuracy"]


class EarlyStoppingCallback:
    """Early stopping callback for training."""
    
    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        monitor: str = "val_loss"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        # Determine if higher is better
        self.mode = "min" if "loss" in monitor.lower() else "max"
    
    def __call__(self, current_score: float, model: Any) -> bool:
        """
        Check if training should stop early.
        
        Args:
            current_score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.model.state_dict().copy()
            return False
        
        # Check if score improved
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.model.load_state_dict(self.best_weights)
                print(f"Restored best weights from {self.patience} epochs ago")
        
        return self.early_stop


class CornerTrainer:
    """Main training class for corner detection."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.model = None
        self.early_stopping = None
        self.lr_scheduler = None
        self.evaluator = None
        
        # Training tracking
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'corner_accuracy_5px': [],
            'corner_accuracy_10px': [],
            'corner_accuracy_20px': [],
            'mAP50': [],
            'mAP50_95': []
        }
        
        self.start_time = None
        self.best_metrics = {}
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Create logs directory
        log_dir = os.path.realpath("logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def setup_directories(self):
        """Setup training directories."""
        self.save_dir = os.path.realpath(self.config.save_dir)
        self.experiment_dir = os.path.join(
            self.save_dir,
            f"corner_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def prepare_data(self, annotations_file: str, images_dir: str) -> str:
        """
        Prepare data for YOLO training format.
        
        Args:
            annotations_file: Path to corner_annotations.json
            images_dir: Base directory containing train/test images
            
        Returns:
            Path to data configuration file
        """
        self.logger.info("Preparing data for YOLO format...")
        
        # Create YOLO format annotations
        yolo_output_dir = os.path.join(self.experiment_dir, "yolo_data")
        create_yolo_annotations(
            annotations_file=annotations_file,
            images_dir=images_dir,
            output_dir=yolo_output_dir,
            image_size=(self.config.img_size, self.config.img_size)
        )
        
        # Create data configuration
        data_config_path = os.path.join(self.experiment_dir, "data.yaml")
        create_data_config(
            train_path=os.path.join(yolo_output_dir, "train"),
            val_path=os.path.join(yolo_output_dir, "test"),
            output_path=data_config_path
        )
        
        self.logger.info(f"Data preparation completed. Config: {data_config_path}")
        return data_config_path
    
    def initialize_model(self):
        """Initialize the corner detection model."""
        self.logger.info("Initializing model...")
        
        self.model = CornerDetectionModel(
            model_name=self.config.model_name,
            num_keypoints=self.config.num_keypoints,
            device=self.config.device
        )
        
        # Initialize evaluator
        self.evaluator = CornerEvaluator()
        
        # Initialize early stopping
        self.early_stopping = EarlyStoppingCallback(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            restore_best_weights=self.config.restore_best_weights,
            monitor="val_loss"
        )
        
        # Log model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model initialized: {model_info}")
    
    def save_config(self):
        """Save training configuration."""
        config_dict = {
            attr: getattr(self.config, attr)
            for attr in dir(self.config)
            if not attr.startswith('_')
        }
        
        config_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Configuration saved: {config_path}")
    
    def train(self, data_config: str):
        """
        Execute training loop.
        
        Args:
            data_config: Path to data configuration file
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.logger.info("Starting training...")
        self.start_time = time.time()
        
        # Training arguments
        train_args = {
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'img_size': self.config.img_size,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'save_dir': self.save_dir,
            'name': os.path.basename(self.experiment_dir),
            'patience': self.config.patience,
            'save_period': self.config.save_period,
            'plots': self.config.plot_results,
            'amp': self.config.mixed_precision,
        }
        
        try:
            # Start training
            results = self.model.train(data_config, **train_args)
            
            # Log training completion
            training_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final results
            self.save_training_results(results)
            
            return results
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            return None
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate_model(self, data_config: str, model_path: Optional[str] = None):
        """
        Validate the trained model.
        
        Args:
            data_config: Path to data configuration file
            model_path: Path to model weights (optional)
        """
        self.logger.info("Starting model validation...")
        
        try:
            results = self.model.validate(
                data_config=data_config,
                model_path=model_path,
                img_size=self.config.img_size,
                batch_size=self.config.batch_size
            )
            
            self.logger.info("Validation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise
    
    def evaluate_corner_accuracy(self, data_config: str, model_path: str):
        """
        Evaluate corner detection accuracy with distance metrics.
        
        Args:
            data_config: Path to data configuration file
            model_path: Path to trained model weights
        """
        self.logger.info("Evaluating corner detection accuracy...")
        
        try:
            # Load data config
            with open(data_config, 'r') as f:
                config = yaml.safe_load(f)
            
            val_dir = config['val']
            
            # Run evaluation
            metrics = self.evaluator.evaluate_model(
                model_path=model_path,
                data_dir=val_dir,
                distance_thresholds=[5, 10, 20]
            )
            
            # Log results
            self.logger.info("Corner Detection Accuracy Results:")
            for threshold, accuracy in metrics['distance_accuracies'].items():
                self.logger.info(f"  Accuracy @ {threshold}px: {accuracy:.3f}")
            
            self.logger.info(f"  Mean distance error: {metrics['mean_distance_error']:.2f}px")
            self.logger.info(f"  Median distance error: {metrics['median_distance_error']:.2f}px")
            
            # Save evaluation results
            eval_path = os.path.join(self.experiment_dir, "evaluation_results.json")
            with open(eval_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Corner accuracy evaluation failed: {str(e)}")
            raise
    
    def save_training_results(self, results: Dict[str, Any]):
        """Save training results and generate plots."""
        # Save results to JSON
        results_path = os.path.join(self.experiment_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate and save plots
        if self.config.plot_results:
            self.generate_training_plots()
        
        self.logger.info(f"Training results saved: {results_path}")
    
    def generate_training_plots(self):
        """Generate training visualization plots."""
        plots_dir = os.path.join(self.experiment_dir, "plots")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Loss curves
        if self.train_history['train_loss'] and self.train_history['val_loss']:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.train_history['epoch'], self.train_history['train_loss'], 'b-', label='Training Loss')
            plt.plot(self.train_history['epoch'], self.train_history['val_loss'], 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.train_history['epoch'], self.train_history['learning_rate'], 'g-')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Corner accuracy plots
        if any(self.train_history[f'corner_accuracy_{px}px'] for px in [5, 10, 20]):
            plt.figure(figsize=(10, 6))
            
            for px in [5, 10, 20]:
                key = f'corner_accuracy_{px}px'
                if self.train_history[key]:
                    plt.plot(self.train_history['epoch'], self.train_history[key], 
                           label=f'Accuracy @ {px}px', linewidth=2)
            
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Corner Detection Accuracy Over Training')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1)
            
            plt.savefig(os.path.join(plots_dir, "corner_accuracy.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Training plots saved to: {plots_dir}")
    
    def run_full_pipeline(self, annotations_file: str, images_dir: str):
        """
        Run the complete training pipeline.
        
        Args:
            annotations_file: Path to corner_annotations.json
            images_dir: Directory containing train/test images
        """
        try:
            # Save configuration
            self.save_config()
            
            # Prepare data
            data_config = self.prepare_data(annotations_file, images_dir)
            
            # Initialize model
            self.initialize_model()
            
            # Train the model
            results = self.train(data_config)
            
            if results is not None:
                # Find best model weights
                best_weights = os.path.join(self.experiment_dir, "training", "weights", "best.pt")
                
                if os.path.exists(best_weights):
                    # Validate the best model
                    self.validate_model(data_config, best_weights)
                    
                    # Evaluate corner detection accuracy
                    self.evaluate_corner_accuracy(data_config, best_weights)
                    
                    self.logger.info(f"Training pipeline completed successfully!")
                    self.logger.info(f"Best model saved at: {best_weights}")
                else:
                    self.logger.warning("Best weights file not found")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train corner detection model")
    
    # Data arguments
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to corner_annotations.json')
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing train/test images')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolo11n-pose.pt',
                       help='YOLO model name or path')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='runs/train',
                       help='Directory to save results')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create training configuration
    config = TrainingConfig()
    
    # Update config with command line arguments
    config.model_name = args.model
    config.img_size = args.img_size
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.patience = args.patience
    config.device = args.device
    config.workers = args.workers
    config.save_dir = args.save_dir
    
    # Validate paths
    annotations_file = os.path.realpath(args.annotations)
    images_dir = os.path.realpath(args.images_dir)
    
    if not os.path.exists(annotations_file):
        print(f"Error: Annotations file not found: {annotations_file}")
        sys.exit(1)
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Create trainer and run pipeline
    trainer = CornerTrainer(config)
    
    print("Starting Magic: The Gathering Corner Detection Training")
    print(f"Annotations: {annotations_file}")
    print(f"Images: {images_dir}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print("-" * 50)
    
    # Run training pipeline
    trainer.run_full_pipeline(annotations_file, images_dir)


if __name__ == "__main__":
    main()