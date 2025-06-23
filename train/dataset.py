"""
Dataset loader for semantic segmentation with data augmentation.
Handles loading images and masks with comprehensive augmentation pipeline.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class CardSegmentationDataset(Dataset):
    """
    Dataset class for card segmentation.
    Loads images and corresponding masks with augmentation support.
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(480, 640)):
        """
        Initialize dataset.
        
        Args:
            image_dir (str): Directory containing input images
            mask_dir (str): Directory containing segmentation masks
            transform (callable, optional): Transform to be applied on samples
            target_size (tuple): Target image size (height, width)
        """
        self.image_dir = os.path.realpath(image_dir)
        self.mask_dir = os.path.realpath(mask_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.image_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing 'image' and 'mask'
        """
        # Get image and mask file paths
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Convert image extension to mask extension (.jpg -> .png)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Warning: Mask not found for {img_name}, creating empty mask")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Convert mask to binary (white pixels = 1, black pixels = 0)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask.long(),
            'filename': img_name
        }

def get_training_transforms(target_size=(480, 640)):
    """
    Get training data augmentation transforms.
    
    Args:
        target_size (tuple): Target image size (height, width)
        
    Returns:
        albumentations.Compose: Composed transforms
    """
    return A.Compose([
        # Random sized crop for scale and crop augmentation
        #A.RandomResizedCrop(
        #    size=(int(target_size[0] * 0.7), int(target_size[1] * 0.7)),
        #    scale=(0.5, 0.9),  # Crop size will be 50-90% of original image
        #    ratio=(0.75, 0.75),  # Aspect ratio will vary from 3:4 to 4:3
        #    interpolation=cv2.INTER_LINEAR,
        #    mask_interpolation=cv2.INTER_NEAREST,
        #    area_for_downscale="image",  # Use INTER_AREA for image downscaling
        #    p=0.4
        #),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.25,
            scale=(0.9, 1.5),
            rotate=(-15, 15),
            mode=cv2.BORDER_CONSTANT,
            cval=0,
            cval_mask=0,
            p=0.8
        ),
        
        # Elastic transform for slight deformation
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),
        
        # Color augmentations (only applied to image)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.8
        ),
        
        # Lighting augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6
        ),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(variance_limit=(10, 50), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.5),
        
        # Random erasing (coarse dropout) for occlusion simulation
        #A.CoarseDropout(
        #    max_holes=3,
        #    max_height=int(target_size[0] * 0.15),
        #    max_width=int(target_size[1] * 0.15),
        #    min_holes=1,
        #    min_height=int(target_size[0] * 0.05),
        #    min_width=int(target_size[1] * 0.05),
        #    fill_value=0,
        #    mask_fill_value=0,
        #    p=0.3
        #),

        # Resize to target size
        A.Resize(height=target_size[0], width=target_size[1]),

        # Normalization and tensor conversion
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_validation_transforms(target_size=(480, 640)):
    """
    Get validation transforms (no augmentation).
    
    Args:
        target_size (tuple): Target image size (height, width)
        
    Returns:
        albumentations.Compose: Composed transforms
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def create_dataloaders(train_image_dir, train_mask_dir, test_image_dir, test_mask_dir,
                      batch_size=8, num_workers=4, pin_memory=True, target_size=(480, 640)):
    """
    Create training and validation data loaders.
    
    Args:
        train_image_dir (str): Training images directory
        train_mask_dir (str): Training masks directory
        test_image_dir (str): Test images directory
        test_mask_dir (str): Test masks directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory
        target_size (tuple): Target image size (height, width)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CardSegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=get_training_transforms(target_size),
        target_size=target_size
    )
    
    val_dataset = CardSegmentationDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        transform=get_validation_transforms(target_size),
        target_size=target_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader

def visualize_sample(dataset, idx=0):
    """
    Visualize a sample from the dataset.
    
    Args:
        dataset (CardSegmentationDataset): Dataset to visualize from
        idx (int): Index of sample to visualize
    """
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    image = sample['image']
    mask = sample['mask']
    filename = sample['filename']
    
    # If image is a tensor, convert to numpy
    if torch.is_tensor(image):
        # Denormalize if normalized
        image = image.permute(1, 2, 0).numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
    
    if torch.is_tensor(mask):
        mask = mask.numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title(f'Image: {filename}')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test dataset loading
    from config import Config
    
    # Create dataset
    dataset = CardSegmentationDataset(
        image_dir=Config.TRAIN_IMAGE_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=get_training_transforms((Config.INPUT_HEIGHT, Config.INPUT_WIDTH))
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading a sample
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Filename: {sample['filename']}")
        
        # Visualize sample (requires matplotlib)
        try:
            visualize_sample(dataset, 0)
        except ImportError:
            print("Matplotlib not available, skipping visualization")
