import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import PIL.Image as Image
from model import generate_gaussian_heatmap
import random
import math


class KeypointTransform:
    """Custom transform class that handles keypoints along with image transformations"""
    
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, image, keypoints):
        # Convert numpy image to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        for transform in self.transforms:
            if hasattr(transform, 'apply_with_keypoints'):
                image, keypoints = transform.apply_with_keypoints(image, keypoints)
            else:
                image = transform(image)
        
        return image, keypoints


class HorizontalFlipWithKeypoints:
    """Horizontal flip that also flips keypoints"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def apply_with_keypoints(self, image, keypoints):
        if random.random() < self.p:
            image = TF.hflip(image)
            width = image.size[0]
            # Flip keypoints horizontally
            keypoints = [(width - x, y) for x, y in keypoints]
        return image, keypoints


class ColorJitterWithKeypoints:
    """Color jitter that preserves keypoints and supports range parameters"""
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0):
        # Convert single values to ranges for compatibility
        self.brightness = self._normalize_param(brightness)
        self.contrast = self._normalize_param(contrast)
        self.saturation = self._normalize_param(saturation)
        self.hue = self._normalize_param(hue)
        self.p = p
    
    def _normalize_param(self, param):
        """Convert parameter to range format"""
        if isinstance(param, (list, tuple)):
            return param
        elif param == 0:
            return None
        else:
            # For TorchVision ColorJitter, ranges should be positive
            return param
    
    def apply_with_keypoints(self, image, keypoints):
        if random.random() < self.p:
            # Create ColorJitter with range parameters directly
            # Convert None to 0 for TorchVision compatibility
            color_jitter = T.ColorJitter(
                brightness=self.brightness if self.brightness is not None else 0,
                contrast=self.contrast if self.contrast is not None else 0,
                saturation=self.saturation if self.saturation is not None else 0,
                hue=self.hue if self.hue is not None else 0
            )
            image = color_jitter(image)
        return image, keypoints


class ResizeWithKeypoints:
    """Resize that scales keypoints accordingly"""
    
    def __init__(self, size):
        self.size = size  # (width, height)
    
    def apply_with_keypoints(self, image, keypoints):
        original_size = image.size  # (width, height)
        image = TF.resize(image, [self.size[1], self.size[0]])  # PIL expects (height, width)
        
        # Scale keypoints
        scale_x = self.size[0] / original_size[0]
        scale_y = self.size[1] / original_size[1]
        keypoints = [(x * scale_x, y * scale_y) for x, y in keypoints]
        
        return image, keypoints


class RandomZoomWithKeypoints:
    """Random zoom augmentation with keypoint adjustment"""
    
    def __init__(self, zoom_range=(0.0, 0.1), p=0.5):
        self.zoom_range = zoom_range  # (min_zoom, max_zoom) as percentage
        self.p = p
    
    def apply_with_keypoints(self, image, keypoints):
        if random.random() < self.p:
            # Get random zoom factor
            zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            
            # Get original dimensions
            width, height = image.size
            
            # Calculate new dimensions (zoom in by cropping)
            new_width = int(width * (1 - zoom_factor))
            new_height = int(height * (1 - zoom_factor))
            
            # Calculate crop coordinates (center crop)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            # Crop the image
            image = image.crop((left, top, right, bottom))
            
            # Resize back to original size
            image = image.resize((width, height), Image.LANCZOS)
            
            # Adjust keypoints
            # First, adjust for the crop
            adjusted_keypoints = []
            for x, y in keypoints:
                # Adjust for crop offset
                new_x = x - left
                new_y = y - top
                
                # Scale to account for resize
                new_x = new_x * width / new_width
                new_y = new_y * height / new_height
                
                adjusted_keypoints.append((new_x, new_y))
            
            keypoints = adjusted_keypoints
        
        return image, keypoints


class RandomRotationWithKeypoints:
    """Random rotation augmentation with keypoint adjustment"""
    
    def __init__(self, degrees=(-5, 5), p=0.5):
        self.degrees = degrees  # (min_degrees, max_degrees)
        self.p = p
    
    def apply_with_keypoints(self, image, keypoints):
        if random.random() < self.p:
            # Get random rotation angle
            angle = random.uniform(self.degrees[0], self.degrees[1])
            
            # Get image center
            width, height = image.size
            center_x, center_y = width / 2, height / 2
            
            # Rotate image
            image = TF.rotate(image, angle, center=(center_x, center_y), fill=0)
            
            # Rotate keypoints (TorchVision rotates clockwise, so negate angle)
            angle_rad = math.radians(-angle)  # Negate for clockwise rotation
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            rotated_keypoints = []
            for x, y in keypoints:
                # Translate to origin
                x_centered = x - center_x
                y_centered = y - center_y
                
                # Apply rotation (clockwise)
                x_rotated = x_centered * cos_angle - y_centered * sin_angle
                y_rotated = x_centered * sin_angle + y_centered * cos_angle
                
                # Translate back
                x_final = x_rotated + center_x
                y_final = y_rotated + center_y
                
                rotated_keypoints.append((x_final, y_final))
            
            keypoints = rotated_keypoints
        
        return image, keypoints


class ToTensorWithKeypoints:
    """Convert PIL image to tensor, preserve keypoints"""
    
    def apply_with_keypoints(self, image, keypoints):
        image = TF.to_tensor(image)
        return image, keypoints


class CornerDataset(Dataset):
    """
    Dataset for corner detection training
    """
    
    def __init__(self, 
                 dataset_path: str,
                 split: str = 'train',
                 transform: Optional[KeypointTransform] = None,
                 image_size: Tuple[int, int] = (480, 640),
                 heatmap_size: Tuple[int, int] = (160, 120),
                 sigma: float = 2.0):
        """
        Initialize dataset
        
        Args:
            dataset_path: Path to dataset directory
            split: 'train' or 'test'
            transform: Augmentation transforms
            image_size: Target image size (width, height)
            heatmap_size: Heatmap size (width, height)
            sigma: Gaussian sigma for heatmap generation
        """
        self.dataset_path = os.path.realpath(dataset_path)
        self.split = split
        self.transform = transform
        self.image_size = image_size  # (width, height)
        self.heatmap_size = heatmap_size  # (width, height)
        self.sigma = sigma
        
        # Load annotations
        annotations_path = os.path.join(self.dataset_path, 'corner_annotations.json')
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Get image list for the split
        self.image_names = list(self.annotations[split].keys())
        
        # Images directory
        self.images_dir = os.path.join(self.dataset_path, split, 'images')
        
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, heatmaps) where heatmaps has shape (4, H, W)
        """
        image_name = self.image_names[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get corners
        corners_data = self.annotations[self.split][image_name]
        
        # Ensure we have exactly 4 corners
        if len(corners_data) != 4:
            print(f"Warning: Image {image_name} has {len(corners_data)} corners, expected 4. Skipping...")
            # Return a dummy sample to maintain batch consistency
            dummy_image = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            dummy_corners = np.array([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9], dtype=np.float32)
            
            if isinstance(dummy_image, np.ndarray):
                dummy_image = torch.from_numpy(np.ascontiguousarray(dummy_image)).permute(2, 0, 1).float() / 255.0
            dummy_corners = torch.from_numpy(np.ascontiguousarray(dummy_corners)).float()
            return dummy_image, dummy_corners
        
        corners = np.array(corners_data, dtype=np.float32)
        
        # Validate corner format
        if corners.shape != (4, 2):
            print(f"Warning: Image {image_name} has invalid corner shape {corners.shape}, expected (4, 2). Skipping...")
            # Return a dummy sample
            dummy_image = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            dummy_corners = np.array([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9], dtype=np.float32)
            
            if isinstance(dummy_image, np.ndarray):
                dummy_image = torch.from_numpy(np.ascontiguousarray(dummy_image)).permute(2, 0, 1).float() / 255.0
            dummy_corners = torch.from_numpy(np.ascontiguousarray(dummy_corners)).float()
            return dummy_image, dummy_corners
        
        # Get original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Create keypoints for transforms
        keypoints = [(corner[0], corner[1]) for corner in corners]
        
        # Apply transforms
        if self.transform:
            image, keypoints = self.transform(image, keypoints)
        
        # Ensure we still have exactly 4 keypoints after transforms
        if len(keypoints) != 4:
            print(f"Warning: Transform resulted in {len(keypoints)} keypoints for {image_name}, expected 4. Using dummy corners.")
            # Use normalized dummy corners for this sample
            keypoints = [(0.1 * self.image_size[0], 0.1 * self.image_size[1]),
                        (0.9 * self.image_size[0], 0.1 * self.image_size[1]),
                        (0.9 * self.image_size[0], 0.9 * self.image_size[1]),
                        (0.1 * self.image_size[0], 0.9 * self.image_size[1])]
        
        # Generate heatmaps for each corner
        heatmaps = np.zeros((4, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        
        for i, (x, y) in enumerate(keypoints):
            # Scale coordinates to heatmap size
            hm_x = x * self.heatmap_size[0] / self.image_size[0]
            hm_y = y * self.heatmap_size[1] / self.image_size[1]
            
            # Generate Gaussian heatmap
            heatmap = generate_gaussian_heatmap(
                center=(hm_x, hm_y),
                heatmap_size=self.heatmap_size,
                sigma=self.sigma
            )
            heatmaps[i] = heatmap
        
        # Convert to tensors (image should already be a tensor if transforms were applied)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        
        heatmaps = torch.from_numpy(np.ascontiguousarray(heatmaps)).float()
        
        return image, heatmaps


def get_train_transforms(image_size: Tuple[int, int] = (480, 640)) -> KeypointTransform:
    """
    Get training transforms with augmentations
    
    Args:
        image_size: Target image size (width, height)
        
    Returns:
        TorchVision compose transform with keypoint support
    """
    return KeypointTransform([
        # Geometric augmentations
        HorizontalFlipWithKeypoints(p=0.5),
        RandomZoomWithKeypoints(zoom_range=(0, 0.3), p=1),
        RandomRotationWithKeypoints(degrees=(-5, 5), p=0.5),
        
        # Color augmentations (only applied to image)
        ColorJitterWithKeypoints(
            brightness=[0.8, 1.2],   # Brightness factor: 0.8 to 1.2
            contrast=[0.8, 1.2],     # Contrast factor: 0.8 to 1.2
            saturation=[0.8, 1.2],   # Saturation factor: 0.8 to 1.2
            hue=[-0.1, 0.1],         # Hue shift: -0.1 to 0.1
            p=0.8
        ),
        
        # Resize to target size
        ResizeWithKeypoints(size=image_size),
        
        # Convert to tensor
        ToTensorWithKeypoints(),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (480, 640)) -> KeypointTransform:
    """
    Get validation transforms (no augmentations)
    
    Args:
        image_size: Target image size (width, height)
        
    Returns:
        TorchVision compose transform with keypoint support
    """
    return KeypointTransform([
        # Resize to target size
        ResizeWithKeypoints(size=image_size),
        
        # Convert to tensor
        ToTensorWithKeypoints(),
    ])


def create_dataloaders(dataset_path: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      image_size: Tuple[int, int] = (480, 640),
                      heatmap_size: Tuple[int, int] = (160, 120)) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        dataset_path: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Target image size (width, height)
        heatmap_size: Heatmap size (width, height)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CornerDataset(
        dataset_path=dataset_path,
        split='train',
        transform=get_train_transforms(image_size),
        image_size=image_size,
        heatmap_size=heatmap_size
    )
    
    val_dataset = CornerDataset(
        dataset_path=dataset_path,
        split='test',  # Use test images for validation
        transform=get_val_transforms(image_size),
        image_size=image_size,
        heatmap_size=heatmap_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def visualize_sample(dataset: CornerDataset, idx: int = 0, save_path: Optional[str] = None):
    """
    Visualize a sample from the dataset
    
    Args:
        dataset: Dataset instance
        idx: Index of sample to visualize
        save_path: Optional path to save the visualization
    """
    image, corners = dataset[idx]
    
    # Convert back to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    
    # Denormalize corners
    corners = corners.numpy()
    corners[0::2] *= dataset.image_size[0]  # x coordinates
    corners[1::2] *= dataset.image_size[1]  # y coordinates
    
    # Draw corners
    image_vis = (image * 255).astype(np.uint8).copy()
    
    # Draw corners as circles
    for i in range(0, len(corners), 2):
        x, y = int(corners[i]), int(corners[i+1])
        cv2.circle(image_vis, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(image_vis, f'{i//2}', (x+10, y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw lines connecting corners
    corner_points = [(int(corners[i]), int(corners[i+1])) for i in range(0, len(corners), 2)]
    if len(corner_points) == 4:
        # Draw rectangle
        cv2.polylines(image_vis, [np.array(corner_points)], True, (0, 255, 0), 2)
    
    if save_path:
        save_path = os.path.realpath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
    
    return image_vis
