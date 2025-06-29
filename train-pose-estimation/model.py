import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator, PosePredictor
from ultralytics.utils import DEFAULT_CFG
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import yaml


class CornerDetectionModel:
    """YOLO pose model wrapper for corner detection."""
    
    def __init__(
        self,
        model_name: str = "yolo12n-pose.pt",
        num_keypoints: int = 4,
        device: str = "auto"
    ):
        """
        Initialize corner detection model.
        
        Args:
            model_name: YOLO model name/path
            num_keypoints: Number of keypoints (4 corners)
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.num_keypoints = num_keypoints
        self.device = self._setup_device(device)
        
        # Initialize YOLO model
        self.model = None
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load YOLO pose model."""
        try:
            # Load pre-trained YOLO pose model
            self.model = YOLO(self.model_name)
            print(f"Loaded model: {self.model_name} on {self.device}")
            
            # Move model to device
            if self.device == "cuda":
                self.model.model.cuda()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_model_config(self, data_config: str) -> str:
        """
        Create custom model configuration for corner detection.
        
        Args:
            data_config: Path to data.yaml configuration
            
        Returns:
            Path to created model configuration
        """
        # Base YOLO11n-pose configuration
        config = {
            # Model backbone
            'backbone': [
                [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
                [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                [-1, 3, 'C2f', [128, True]],
                [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                [-1, 6, 'C2f', [256, True]],
                [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                [-1, 6, 'C2f', [512, True]],
                [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
                [-1, 3, 'C2f', [1024, True]],
                [-1, 1, 'SPPF', [1024, 5]],  # 9
            ],
            
            # Head
            'head': [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                [-1, 3, 'C2f', [512]],  # 12
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                [-1, 3, 'C2f', [256]],  # 15 (P3/8-small)
                
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 12], 1, 'Concat', [1]],  # cat head P4
                [-1, 3, 'C2f', [512]],  # 18 (P4/16-medium)
                
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 9], 1, 'Concat', [1]],  # cat head P5
                [-1, 3, 'C2f', [1024]],  # 21 (P5/32-large)
                
                [[15, 18, 21], 1, 'Pose', [1, self.num_keypoints]],  # Pose head
            ]
        }
        
        # Save configuration
        config_path = os.path.realpath("corner_detection_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def train(
        self,
        data_config: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0005,
        save_dir: str = "runs/train",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the corner detection model.
        
        Args:
            data_config: Path to data.yaml configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_dir: Directory to save training results
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Set up training arguments
        train_args = {
            'data': os.path.realpath(data_config),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': learning_rate,
            'weight_decay': weight_decay,
            'device': self.device,
            'project': os.path.realpath(save_dir),
            'name': kwargs.get('name', 'corner_detection'),
            'exist_ok': True,
            'patience': kwargs.get('patience', 20),
            'save_period': kwargs.get('save_period', 10),
            'val': True,
            'plots': kwargs.get('plots', True),
            'amp': kwargs.get('amp', True),
            **{k: v for k, v in kwargs.items() if k not in ['name', 'patience', 'save_period', 'plots', 'amp']}
        }
        
        # Start training
        print(f"Starting training with {epochs} epochs...")
        results = self.model.train(**train_args)
        
        print("Training completed!")
        return results
    
    def validate(
        self,
        data_config: str,
        model_path: Optional[str] = None,
        img_size: int = 640,
        batch_size: int = 16,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate the model.
        
        Args:
            data_config: Path to data.yaml configuration
            model_path: Path to model weights (if None, uses current model)
            img_size: Input image size
            batch_size: Validation batch size
            **kwargs: Additional validation arguments
            
        Returns:
            Validation results dictionary
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for validation")
        
        val_args = {
            'data': os.path.realpath(data_config),
            'imgsz': img_size,
            'batch': batch_size,
            'device': self.device,
            'save_json': True,
            'plots': True,
            **kwargs
        }
        
        print("Starting validation...")
        results = model.val(**val_args)
        
        print("Validation completed!")
        return results
    
    def predict(
        self,
        source: str,
        model_path: Optional[str] = None,
        img_size: int = 640,
        conf_threshold: float = 0.25,
        save_results: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on images.
        
        Args:
            source: Path to image(s) or directory
            model_path: Path to model weights (if None, uses current model)
            img_size: Input image size
            conf_threshold: Confidence threshold
            save_results: Whether to save prediction results
            **kwargs: Additional prediction arguments
            
        Returns:
            List of prediction results
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for prediction")
        
        pred_args = {
            'source': source,
            'imgsz': img_size,
            'conf': conf_threshold,
            'device': self.device,
            'save': save_results,
            'save_txt': save_results,
            'save_conf': True,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'max_det': 1,  # Only detect one card per image
            **kwargs
        }
        
        print(f"Running inference on: {source}")
        results = model.predict(**pred_args)
        
        return results
    
    def export_onnx(
        self,
        model_path: str,
        output_path: str,
        img_size: int = 640,
        dynamic: bool = True,
        half: bool = False,
        **kwargs
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            model_path: Path to trained model weights
            output_path: Output path for ONNX model
            img_size: Input image size
            dynamic: Enable dynamic input shapes
            half: Use half precision
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported ONNX model
        """
        model = YOLO(model_path)
        
        export_args = {
            'format': 'onnx',
            'imgsz': img_size,
            'dynamic': dynamic,
            'half': half,
            'opset': 11,
            'simplify': True,
            **kwargs
        }
        
        print(f"Exporting model to ONNX: {output_path}")
        exported_path = model.export(**export_args)
        
        # Move to desired location if different
        if output_path != exported_path:
            os.rename(exported_path, output_path)
            exported_path = output_path
        
        print(f"Model exported successfully: {exported_path}")
        return exported_path
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "num_keypoints": self.num_keypoints,
            "parameters": sum(p.numel() for p in self.model.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        }
        
        return info
    
    def load_weights(self, weights_path: str):
        """Load model weights."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.model = YOLO(weights_path)
        print(f"Loaded weights from: {weights_path}")
    
    def save_weights(self, output_path: str):
        """Save model weights."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save(self.model.model.state_dict(), output_path)
        print(f"Saved weights to: {output_path}")


def create_data_config(
    train_path: str,
    val_path: str,
    output_path: str = "data.yaml"
) -> str:
    """
    Create YOLO data configuration file.
    
    Args:
        train_path: Path to training data directory
        val_path: Path to validation data directory
        output_path: Output path for data.yaml
        
    Returns:
        Path to created configuration file
    """
    config = {
        'train': os.path.realpath(train_path),
        'val': os.path.realpath(val_path),
        'nc': 1,  # Number of classes
        'names': ['card'],  # Class names
        'kpt_shape': [4, 3],  # 4 keypoints, each with (x, y, visibility)
        'flip_idx': [1, 0, 3, 2]  # Horizontal flip mapping for keypoints
    }
    
    output_path = os.path.realpath(output_path)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Data configuration saved to: {output_path}")
    return output_path


class CustomPoseTrainer(PoseTrainer):
    """Custom trainer with additional features for corner detection."""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)
        self.best_corner_accuracy = 0.0
    
    def validate(self):
        """Custom validation with corner-specific metrics."""
        results = super().validate()
        
        # Add custom corner detection metrics here if needed
        # For example, calculate percentage of corners within certain pixel distances
        
        return results


if __name__ == "__main__":
    # Example usage
    print("Testing Corner Detection Model...")
    
    # Create model instance
    model = CornerDetectionModel(
        model_name="yolo11n-pose.pt",
        num_keypoints=4,
        device="auto"
    )
    
    # Print model information
    info = model.get_model_info()
    print(f"Model Info: {info}")
    
    # Create data configuration
    data_config = create_data_config(
        train_path="../dataset/yolo_format/train",
        val_path="../dataset/yolo_format/test",
        output_path="corner_data.yaml"
    )
    
    print("Model setup complete!")