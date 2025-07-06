#!/usr/bin/env python3
"""
Script to visualize training augmentations
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from dataset import CornerDataset, get_train_transforms, get_val_transforms
import argparse


def draw_keypoints_on_image(image, keypoints, color=(0, 255, 0)):
    """
    Draw keypoints on image
    
    Args:
        image: Image as numpy array (H, W, 3)
        keypoints: List of (x, y) keypoints
        color: Color for keypoints (B, G, R)
    
    Returns:
        Image with keypoints drawn
    """
    image_vis = image.copy()
    
    # Draw keypoints as circles
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(image_vis, (int(x), int(y)), 5, color, -1)
        cv2.putText(image_vis, f'{i}', (int(x)+10, int(y)+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw lines connecting keypoints (assuming 4 corners)
    if len(keypoints) == 4:
        corner_points = np.array([(int(x), int(y)) for x, y in keypoints])
        cv2.polylines(image_vis, [corner_points], True, color, 2)
    
    return image_vis


def tensor_to_numpy(tensor_image):
    """Convert tensor image to numpy array"""
    if isinstance(tensor_image, torch.Tensor):
        # Convert from (C, H, W) to (H, W, C)
        image = tensor_image.permute(1, 2, 0).numpy()
        # Convert from [0,1] to [0,255]
        image = (image * 255).astype(np.uint8)
    else:
        image = np.array(tensor_image)
    return image


def visualize_augmentations(dataset_path, num_samples=5, save_dir="augmentation_visualizations"):
    """
    Visualize training augmentations
    
    Args:
        dataset_path: Path to dataset
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    # Dataset without augmentations (validation transforms)
    val_transforms = get_val_transforms(image_size=(480, 640))
    dataset_original = CornerDataset(
        dataset_path=dataset_path,
        split='train',
        transform=val_transforms,
        image_size=(480, 640),
        heatmap_size=(160, 120)
    )
    
    # Dataset with augmentations (training transforms)
    train_transforms = get_train_transforms(image_size=(480, 640))
    dataset_augmented = CornerDataset(
        dataset_path=dataset_path,
        split='train',
        transform=train_transforms,
        image_size=(480, 640),
        heatmap_size=(160, 120)
    )
    
    print(f"Visualizing {num_samples} samples with augmentations...")
    
    for i in range(min(num_samples, len(dataset_original))):
        # Get original and augmented versions
        original_image, original_heatmaps = dataset_original[i]
        augmented_image, augmented_heatmaps = dataset_augmented[i]
        
        # Convert tensors to numpy
        original_np = tensor_to_numpy(original_image)
        augmented_np = tensor_to_numpy(augmented_image)
        
        # Get keypoints from heatmaps (find peaks)
        original_keypoints = []
        augmented_keypoints = []
        
        for j in range(4):  # 4 corners
            # Original keypoints
            heatmap = original_heatmaps[j].numpy()
            y_orig, x_orig = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            # Scale to image size
            x_orig = x_orig * original_np.shape[1] / heatmap.shape[1]
            y_orig = y_orig * original_np.shape[0] / heatmap.shape[0]
            original_keypoints.append((x_orig, y_orig))
            
            # Augmented keypoints
            heatmap = augmented_heatmaps[j].numpy()
            y_aug, x_aug = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            # Scale to image size
            x_aug = x_aug * augmented_np.shape[1] / heatmap.shape[1]
            y_aug = y_aug * augmented_np.shape[0] / heatmap.shape[0]
            augmented_keypoints.append((x_aug, y_aug))
        
        # Draw keypoints on images
        original_vis = draw_keypoints_on_image(original_np, original_keypoints, color=(0, 255, 0))
        augmented_vis = draw_keypoints_on_image(augmented_np, augmented_keypoints, color=(0, 0, 255))
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original (Validation Transforms)', fontsize=14)
        axes[0].axis('off')
        
        # Augmented image
        axes[1].imshow(cv2.cvtColor(augmented_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Augmented (Training Transforms)', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save the comparison
        save_path = os.path.join(save_dir, f'augmentation_comparison_{i:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization {i+1}/{num_samples}: {save_path}")


def visualize_individual_augmentations(dataset_path, sample_idx=0, save_dir="individual_augmentations"):
    """
    Visualize individual augmentation effects
    
    Args:
        dataset_path: Path to dataset
        sample_idx: Index of sample to use
        save_dir: Directory to save visualizations
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataset without transforms to get raw data
    dataset_raw = CornerDataset(
        dataset_path=dataset_path,
        split='train',
        transform=None,
        image_size=(480, 640),
        heatmap_size=(160, 120)
    )
    
    # Get raw sample
    raw_image, _ = dataset_raw[sample_idx]
    
    # Convert to numpy and get keypoints from annotations
    if isinstance(raw_image, torch.Tensor):
        raw_np = tensor_to_numpy(raw_image)
    else:
        raw_np = raw_image
    
    # Get keypoints from dataset annotations
    image_name = dataset_raw.image_names[sample_idx]
    corners_data = dataset_raw.annotations[dataset_raw.split][image_name]
    keypoints = [(corner[0], corner[1]) for corner in corners_data]
    
    # Convert to PIL for transforms
    raw_pil = Image.fromarray(raw_np)
    
    # Test individual augmentations
    from dataset import (HorizontalFlipWithKeypoints, RandomZoomWithKeypoints, 
                        ColorJitterWithKeypoints, ResizeWithKeypoints)
    
    augmentations = [
        ("Original", None),
        ("Horizontal Flip", HorizontalFlipWithKeypoints(p=1.0)),
        ("Random Zoom", RandomZoomWithKeypoints(zoom_range=(0.05, 0.1), p=1.0)),
        ("Color Jitter", ColorJitterWithKeypoints(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)),
        ("Resize", ResizeWithKeypoints(size=(480, 640))),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, transform) in enumerate(augmentations):
        if i >= len(axes):
            break
            
        if transform is None:
            # Original image
            image_vis = draw_keypoints_on_image(raw_np, keypoints, color=(0, 255, 0))
        else:
            # Apply transform
            transformed_image, transformed_keypoints = transform.apply_with_keypoints(raw_pil, keypoints)
            
            # Convert back to numpy
            if isinstance(transformed_image, torch.Tensor):
                transformed_np = tensor_to_numpy(transformed_image)
            else:
                transformed_np = np.array(transformed_image)
            
            image_vis = draw_keypoints_on_image(transformed_np, transformed_keypoints, color=(0, 0, 255))
        
        axes[i].imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(name, fontsize=12)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(augmentations), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    save_path = os.path.join(save_dir, f'individual_augmentations_sample_{sample_idx:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual augmentations: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training augmentations")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to visualize")
    parser.add_argument("--save-dir", type=str, default="augmentation_visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--individual", action="store_true",
                       help="Also create individual augmentation visualizations")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Sample index for individual augmentation visualization")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist")
        return
    
    print("Creating augmentation visualizations...")
    
    # Create comparison visualizations
    visualize_augmentations(
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
    
    # Create individual augmentation visualizations if requested
    if args.individual:
        individual_dir = os.path.join(args.save_dir, "individual")
        visualize_individual_augmentations(
            dataset_path=args.dataset_path,
            sample_idx=args.sample_idx,
            save_dir=individual_dir
        )
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()