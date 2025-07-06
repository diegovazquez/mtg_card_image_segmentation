import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import timm
import numpy as np


class HRNetPoseHead(nn.Module):
    """
    HRNet pose estimation head for heatmap-based keypoint detection
    Based on https://github.com/HRNet/HRNet-Human-Pose-Estimation
    """
    
    def __init__(self, in_channels: int, num_keypoints: int = 4, target_size: Tuple[int, int] = (160, 120)):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.target_size = target_size  # (width, height)
        
        # Adaptive approach: use deconv layers + adaptive pooling to match target size
        self.deconv_layers = nn.ModuleList()
        self.deconv_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        )
        self.deconv_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        )
        
        # Intermediate conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final layer to generate heatmaps
        self.final_layer = nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1, padding=0)
        
        # Adaptive pooling to ensure exact target size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(target_size[::-1])  # (height, width)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate heatmaps
        
        Args:
            x: Input feature tensor
            
        Returns:
            Heatmaps of shape (B, num_keypoints, target_height, target_width)
        """
        # Apply deconvolution layers
        for deconv in self.deconv_layers:
            x = deconv(x)
        
        # Apply intermediate conv layers
        x = self.conv_layers(x)
        
        # Generate heatmaps
        heatmaps = self.final_layer(x)
        
        # Resize to exact target size
        heatmaps = self.adaptive_pool(heatmaps)
        
        return heatmaps


class LiteHRNet(nn.Module):
    """
    Lite-HRNet model for corner detection using pose estimation approach
    Based on https://github.com/HRNet/HRNet-Human-Pose-Estimation
    """
    
    def __init__(self, num_keypoints: int = 4, heatmap_size: Tuple[int, int] = (160, 120)):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size  # (width, height)
        
        # Use timm's HRNet as backbone
        self.backbone = timm.create_model(
            'hrnet_w18_small',
            pretrained=True,
            features_only=True,
            out_indices=[3]  # Use the highest resolution feature map
        )
        
        # Get the feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 480)
            features = self.backbone(dummy_input)
            self.feature_dim = features[0].shape[1]
            feature_h, feature_w = features[0].shape[2], features[0].shape[3]
            print(f"Backbone feature size: {feature_h}x{feature_w}")
            
            # Calculate output size after deconv layers
            # Each deconv layer doubles the size with stride=2
            output_h = feature_h * (2 ** 3)  # 3 deconv layers
            output_w = feature_w * (2 ** 3)
            print(f"Expected heatmap output size: {output_h}x{output_w}")
            print(f"Target heatmap size: {self.heatmap_size[1]}x{self.heatmap_size[0]}")
        
        # HRNet pose estimation head
        self.pose_head = HRNetPoseHead(self.feature_dim, num_keypoints, self.heatmap_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, 3, 640, 480)
        Returns:
            Heatmaps tensor of shape (B, num_keypoints, H, W)
        """
        # Extract features
        features = self.backbone(x)[0]  # Get the highest resolution features
        
        # Generate heatmaps
        heatmaps = self.pose_head(features)
        
        return heatmaps
    
    def decode_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Decode heatmaps to keypoint coordinates
        
        Args:
            heatmaps: Heatmaps of shape (B, num_keypoints, H, W)
            
        Returns:
            Coordinates tensor of shape (B, num_keypoints * 2)
        """
        batch_size, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape
        
        # Reshape heatmaps to (B, num_keypoints, H*W)
        heatmaps_reshaped = heatmaps.view(batch_size, num_keypoints, -1)
        
        # Find maximum locations
        max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
        
        # Convert indices to coordinates
        y_coords = (max_indices // heatmap_width).float()
        x_coords = (max_indices % heatmap_width).float()
        
        # Normalize coordinates to [0, 1]
        x_coords = x_coords / (heatmap_width - 1)
        y_coords = y_coords / (heatmap_height - 1)
        
        # Interleave x and y coordinates
        coords = torch.zeros(batch_size, num_keypoints * 2, device=heatmaps.device)
        coords[:, 0::2] = x_coords  # x coordinates
        coords[:, 1::2] = y_coords  # y coordinates
        
        return coords


def generate_gaussian_heatmap(center: Tuple[float, float], 
                            heatmap_size: Tuple[int, int],
                            sigma: float = 2.0) -> np.ndarray:
    """
    Generate a Gaussian heatmap for a keypoint
    
    Args:
        center: (x, y) coordinates of the keypoint center
        heatmap_size: (width, height) of the heatmap
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Gaussian heatmap as numpy array
    """
    width, height = heatmap_size
    x_center, y_center = center
    
    # Create coordinate grids
    x = np.arange(0, width, dtype=np.float32)
    y = np.arange(0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    
    # Generate Gaussian heatmap
    heatmap = np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2))
    
    return heatmap


def create_model(num_keypoints: int = 4, pretrained: bool = True, heatmap_size: Tuple[int, int] = (160, 120)) -> LiteHRNet:
    """
    Create Lite-HRNet model for corner detection with pose estimation head
    
    Args:
        num_keypoints: Number of keypoints (4 for 4 corners)
        pretrained: Whether to use pretrained backbone
        heatmap_size: Heatmap output size (width, height)
        
    Returns:
        LiteHRNet model
    """
    try:
        model = LiteHRNet(num_keypoints=num_keypoints, heatmap_size=heatmap_size)
        return model
    except Exception as e:
        print(f"Warning: Model creation encountered issues: {e}")
        print("This may be expected if model is being adapted for corner detection.")
        model = LiteHRNet(num_keypoints=num_keypoints, heatmap_size=heatmap_size)
        return model


def load_model(checkpoint_path: str, num_keypoints: int = 4, heatmap_size: Tuple[int, int] = (160, 120)) -> LiteHRNet:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_keypoints: Number of keypoints
        heatmap_size: Heatmap output size (width, height)
        
    Returns:
        Loaded model
    """
    model = create_model(num_keypoints=num_keypoints, heatmap_size=heatmap_size)
    
    checkpoint_path = os.path.realpath(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def save_model(model: nn.Module, checkpoint_path: str, 
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: int = 0, loss: float = 0.0, metrics: dict = None):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
    """
    checkpoint_path = os.path.realpath(checkpoint_path)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics or {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path)