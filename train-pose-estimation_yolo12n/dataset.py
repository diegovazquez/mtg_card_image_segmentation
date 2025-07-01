import json
import os
import shutil
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CornerDataset(Dataset):
    """Dataset class for Magic: The Gathering card corner detection using YOLO pose format."""
    
    def __init__(
        self,
        annotations_file: str,
        images_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (480, 640),
        augmentations: bool = True
    ):
        """
        Initialize the corner detection dataset.
        
        Args:
            annotations_file: Path to corner_annotations.json
            images_dir: Base directory containing train/test image folders
            split: Either 'train' or 'test'
            image_size: Target image size (width, height)
            augmentations: Whether to apply data augmentations
        """
        self.annotations_file = os.path.realpath(annotations_file)
        self.images_dir = os.path.realpath(images_dir)
        self.split = split
        self.image_size = image_size
        self.augmentations = augmentations
        
        # Load annotations
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get image list for the specified split
        if split in self.annotations:
            self.image_names = list(self.annotations[split].keys())
        else:
            raise ValueError(f"Split '{split}' not found in annotations file")
        
        # Setup transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transforms and augmentations."""
        if self.augmentations and self.split == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.Resize(height=self.image_size[1], width=self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size[1], width=self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image, keypoints, and metadata
        """
        image_name = self.image_names[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, self.split, "images", image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get corner annotations (4 points: top-left, top-right, bottom-right, bottom-left)
        corners = self.annotations[self.split][image_name]
        keypoints = [(float(x), float(y)) for x, y in corners]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints']
        
        # Convert keypoints to YOLO pose format
        # Normalize keypoints to [0, 1] range
        normalized_keypoints = []
        for x, y in keypoints:
            norm_x = x / self.image_size[0]
            norm_y = y / self.image_size[1]
            normalized_keypoints.extend([norm_x, norm_y, 2])  # 2 = visible
        
        # Create bounding box from keypoints for YOLO format
        xs = [kp[0] for kp in keypoints]
        ys = [kp[1] for kp in keypoints]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add padding to bbox
        padding = 0.05
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        bbox_width += bbox_width * padding
        bbox_height += bbox_height * padding
        
        # Normalize bbox coordinates
        norm_x_center = x_center / self.image_size[0]
        norm_y_center = y_center / self.image_size[1]
        norm_width = bbox_width / self.image_size[0]
        norm_height = bbox_height / self.image_size[1]
        
        # YOLO pose format: [class_id, x_center, y_center, width, height, kp1_x, kp1_y, kp1_v, ...]
        yolo_label = [0, norm_x_center, norm_y_center, norm_width, norm_height] + normalized_keypoints
        
        return {
            'image': image,
            'label': torch.tensor(yolo_label, dtype=torch.float32),
            'image_name': image_name,
            'original_size': (original_width, original_height)
        }
    
    def get_class_names(self) -> List[str]:
        """Return class names for the dataset."""
        return ['card']
    
    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        return {
            'num_samples': len(self.image_names),
            'num_classes': 1,
            'image_size': self.image_size,
            'split': self.split
        }


def create_yolo_annotations(
    annotations_file: str,
    images_dir: str,
    output_dir: str,
    image_size: Tuple[int, int] = (480, 640)
):
    """
    Create YOLO format annotation files from corner_annotations.json.
    
    Args:
        annotations_file: Path to corner_annotations.json
        images_dir: Base directory containing train/test image folders
        output_dir: Output directory for YOLO format files
        image_size: Target image size (width, height)
    """
    annotations_file = os.path.realpath(annotations_file)
    images_dir = os.path.realpath(images_dir)
    output_dir = os.path.realpath(output_dir)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    for split in ['train', 'test']:
        if split not in annotations:
            continue
            
        print(f"Processing {split} split...")
        
        for image_name, corners in annotations[split].items():
            # Load image to get original dimensions
            image_path = os.path.join(images_dir, split, "images", image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Image not found: {image_path}")
                continue
            
            original_height, original_width = image.shape[:2]
            
            # Scale corners to target image size
            scale_x = image_size[0] / original_width
            scale_y = image_size[1] / original_height
            
            scaled_corners = []
            for x, y in corners:
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                scaled_corners.append((scaled_x, scaled_y))
            
            # Create bounding box
            xs = [corner[0] for corner in scaled_corners]
            ys = [corner[1] for corner in scaled_corners]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Add padding to bbox
            padding = 0.05
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            bbox_width += bbox_width * padding
            bbox_height += bbox_height * padding
            
            # Normalize coordinates
            norm_x_center = x_center / image_size[0]
            norm_y_center = y_center / image_size[1]
            norm_width = bbox_width / image_size[0]
            norm_height = bbox_height / image_size[1]
            
            # Normalize keypoints
            normalized_keypoints = []
            for x, y in scaled_corners:
                norm_x = x / image_size[0]
                norm_y = y / image_size[1]
                normalized_keypoints.extend([norm_x, norm_y, 2])  # 2 = visible
            
            # Create YOLO label
            yolo_label = [0, norm_x_center, norm_y_center, norm_width, norm_height] + normalized_keypoints
            
            # Save annotation file
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(output_dir, split, 'labels', label_name)
            
            with open(label_path, 'w') as f:
                f.write(' '.join(map(str, yolo_label)) + '\n')
            
            # Copy image to YOLO format directory
            src_image_path = os.path.join(images_dir, split, "images", image_name)
            dst_image_path = os.path.join(output_dir, split, 'images', image_name)
            
            try:
                shutil.copy2(src_image_path, dst_image_path)
            except Exception as e:
                print(f"Warning: Failed to copy image {src_image_path}: {e}")
    
    # Create data.yaml file
    yaml_content = f"""
train: {os.path.join(output_dir, 'train')}
val: {os.path.join(output_dir, 'test')}

nc: 1
names: ['card']

# Keypoint configuration for corner detection
kpt_shape: [4, 3]  # 4 keypoints, each with (x, y, visibility)
flip_idx: [1, 0, 3, 2]  # Horizontal flip mapping: TL->TR, TR->TL, BR->BL, BL->BR
"""
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"YOLO annotations created in: {output_dir}")


if __name__ == "__main__":
    # Example usage
    dataset_root = os.path.realpath("../dataset")
    annotations_file = os.path.join(dataset_root, "corner_annotations.json")
    images_dir = dataset_root
    
    # Create train dataset
    train_dataset = CornerDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        split="train",
        augmentations=True
    )
    
    # Create test dataset
    test_dataset = CornerDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        split="test",
        augmentations=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample label shape: {sample['label'].shape}")
    print(f"Sample label: {sample['label']}")
    
    # Create YOLO format annotations
    create_yolo_annotations(
        annotations_file=annotations_file,
        images_dir=images_dir,
        output_dir=os.path.join(dataset_root, "yolo_format")
    )