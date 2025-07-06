import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.nn as nn


class CornerMetrics:
    """
    Custom metrics for corner detection evaluation
    """
    
    def __init__(self, image_size: Tuple[int, int] = (480, 640)):
        """
        Initialize metrics
        
        Args:
            image_size: Image size (width, height) for denormalization
        """
        self.image_size = image_size
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_distances = []
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions and targets
        
        Args:
            predictions: Predicted heatmaps of shape (B, 4, H, W)
            targets: Target heatmaps of shape (B, 4, H, W)
        """
        from model import LiteHRNet
        
        # Decode heatmaps to coordinates
        batch_size, num_keypoints, heatmap_height, heatmap_width = predictions.shape
        
        # Reshape heatmaps to (B, num_keypoints, H*W)
        pred_reshaped = predictions.view(batch_size, num_keypoints, -1)
        target_reshaped = targets.view(batch_size, num_keypoints, -1)
        
        # Find maximum locations for predictions
        pred_max_vals, pred_max_indices = torch.max(pred_reshaped, dim=2)
        target_max_vals, target_max_indices = torch.max(target_reshaped, dim=2)
        
        # Convert indices to coordinates
        pred_y_coords = (pred_max_indices // heatmap_width).float()
        pred_x_coords = (pred_max_indices % heatmap_width).float()
        target_y_coords = (target_max_indices // heatmap_width).float()
        target_x_coords = (target_max_indices % heatmap_width).float()
        
        # Scale coordinates to image size
        pred_x_coords = pred_x_coords * self.image_size[0] / (heatmap_width - 1)
        pred_y_coords = pred_y_coords * self.image_size[1] / (heatmap_height - 1)
        target_x_coords = target_x_coords * self.image_size[0] / (heatmap_width - 1)
        target_y_coords = target_y_coords * self.image_size[1] / (heatmap_height - 1)
        
        # Convert to numpy
        pred_x_np = pred_x_coords.detach().cpu().numpy()
        pred_y_np = pred_y_coords.detach().cpu().numpy()
        target_x_np = target_x_coords.detach().cpu().numpy()
        target_y_np = target_y_coords.detach().cpu().numpy()
        
        # Calculate distances for each corner
        for i in range(batch_size):  # For each sample in batch
            for j in range(num_keypoints):  # For each corner
                distance = np.sqrt((pred_x_np[i, j] - target_x_np[i, j]) ** 2 + 
                                 (pred_y_np[i, j] - target_y_np[i, j]) ** 2)
                self.all_distances.append(distance)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.all_distances:
            return {
                'corner_acc_3px': 0.0,
                'corner_acc_6px': 0.0,
                'mean_corner_distance': 0.0
            }
        
        distances = np.array(self.all_distances)
        
        # Corner accuracy metrics
        corner_acc_3px = np.mean(distances <= 3.0) * 100
        corner_acc_6px = np.mean(distances <= 6.0) * 100
        
        # Mean corner distance
        mean_corner_distance = np.mean(distances)
        
        return {
            'corner_acc_3px': corner_acc_3px,
            'corner_acc_6px': corner_acc_6px,
            'mean_corner_distance': mean_corner_distance
        }


class CornerLoss(nn.Module):
    """
    Heatmap-based loss for corner detection with pose estimation approach
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (480, 640),
                 heatmap_size: Tuple[int, int] = (160, 120)):
        """
        Initialize loss function
        
        Args:
            image_size: Image size (width, height)
            heatmap_size: Heatmap size (width, height)
        """
        super().__init__()
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of loss function using heatmap MSE
        
        Args:
            predictions: Predicted heatmaps of shape (B, 4, H, W)
            targets: Target heatmaps of shape (B, 4, H, W)
            
        Returns:
            Heatmap MSE loss
        """
        return self.mse_loss(predictions, targets)


def calculate_corner_accuracy(predictions: torch.Tensor, 
                            targets: torch.Tensor,
                            threshold_px: float = 3.0,
                            image_size: Tuple[int, int] = (480, 640)) -> float:
    """
    Calculate corner accuracy within a pixel threshold
    
    Args:
        predictions: Predicted corners of shape (B, 8)
        targets: Target corners of shape (B, 8)
        threshold_px: Pixel threshold for accuracy
        image_size: Image size (width, height)
        
    Returns:
        Accuracy percentage
    """
    # Convert to numpy and denormalize
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Denormalize coordinates
    pred_np[:, 0::2] *= image_size[0]  # x coordinates
    pred_np[:, 1::2] *= image_size[1]  # y coordinates
    target_np[:, 0::2] *= image_size[0]  # x coordinates
    target_np[:, 1::2] *= image_size[1]  # y coordinates
    
    # Calculate distances
    distances = []
    for i in range(pred_np.shape[0]):
        pred_corners = pred_np[i].reshape(-1, 2)
        target_corners = target_np[i].reshape(-1, 2)
        corner_distances = np.sqrt(np.sum((pred_corners - target_corners) ** 2, axis=1))
        distances.extend(corner_distances)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(distances) <= threshold_px) * 100
    return accuracy


def calculate_mean_corner_distance(predictions: torch.Tensor,
                                 targets: torch.Tensor,
                                 image_size: Tuple[int, int] = (480, 640)) -> float:
    """
    Calculate mean corner distance in pixels
    
    Args:
        predictions: Predicted corners of shape (B, 8)
        targets: Target corners of shape (B, 8)
        image_size: Image size (width, height)
        
    Returns:
        Mean distance in pixels
    """
    # Convert to numpy and denormalize
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Denormalize coordinates
    pred_np[:, 0::2] *= image_size[0]  # x coordinates
    pred_np[:, 1::2] *= image_size[1]  # y coordinates
    target_np[:, 0::2] *= image_size[0]  # x coordinates
    target_np[:, 1::2] *= image_size[1]  # y coordinates
    
    # Calculate distances
    distances = []
    for i in range(pred_np.shape[0]):
        pred_corners = pred_np[i].reshape(-1, 2)
        target_corners = target_np[i].reshape(-1, 2)
        corner_distances = np.sqrt(np.sum((pred_corners - target_corners) ** 2, axis=1))
        distances.extend(corner_distances)
    
    return np.mean(distances)


class EarlyStopping:
    """
    Early stopping utility
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - min_delta
        else:
            self.monitor_op = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current score to monitor
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            return False
        
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False