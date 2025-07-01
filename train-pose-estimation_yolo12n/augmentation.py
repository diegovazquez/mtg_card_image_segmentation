import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import random


class CornerAugmentation:
    """Advanced augmentation pipeline for corner detection training."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (480, 640),
        augmentation_probability: float = 0.8
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size (width, height)
            augmentation_probability: Probability of applying augmentations
        """
        self.image_size = image_size
        self.augmentation_probability = augmentation_probability
        
        # Define augmentation pipelines
        self.light_augmentations = self._create_light_augmentations()
        self.medium_augmentations = self._create_medium_augmentations()
        self.heavy_augmentations = self._create_heavy_augmentations()
        self.geometric_augmentations = self._create_geometric_augmentations()
        self.color_augmentations = self._create_color_augmentations()
        
        # Validation transforms (no augmentation)
        self.validation_transforms = A.Compose([
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _create_light_augmentations(self) -> A.Compose:
        """Create light augmentation pipeline for early training stages."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _create_medium_augmentations(self) -> A.Compose:
        """Create medium augmentation pipeline for regular training."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=15,
                val_shift_limit=8,
                p=0.3
            ),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _create_heavy_augmentations(self) -> A.Compose:
        """Create heavy augmentation pipeline for robust training."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=20, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=12,
                sat_shift_limit=20,
                val_shift_limit=12,
                p=0.4
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5))
            ], p=0.3),
            A.OneOf([
                A.Blur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5)
            ], p=0.3),
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _create_geometric_augmentations(self) -> A.Compose:
        """Create geometric-focused augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=25, p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.6
            ),
            A.Perspective(scale=(0.05, 0.15), p=0.3),
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=10,
                p=0.2
            ),
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _create_color_augmentations(self) -> A.Compose:
        """Create color-focused augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.6
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.4
            ),
            A.ChannelShuffle(p=0.1),
            A.ToGray(p=0.1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def get_transforms(self, mode: str = "medium") -> A.Compose:
        """
        Get augmentation transforms based on mode.
        
        Args:
            mode: Augmentation mode ('light', 'medium', 'heavy', 'geometric', 'color', 'validation')
            
        Returns:
            Albumentations compose object
        """
        transforms_map = {
            'light': self.light_augmentations,
            'medium': self.medium_augmentations,
            'heavy': self.heavy_augmentations,
            'geometric': self.geometric_augmentations,
            'color': self.color_augmentations,
            'validation': self.validation_transforms
        }
        
        if mode not in transforms_map:
            raise ValueError(f"Unknown augmentation mode: {mode}")
        
        return transforms_map[mode]
    
    def apply_random_augmentation(
        self,
        image: np.ndarray,
        keypoints: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Apply random augmentation from different pipelines.
        
        Args:
            image: Input image
            keypoints: List of keypoints as (x, y) tuples
            
        Returns:
            Dictionary with augmented image and keypoints
        """
        if random.random() > self.augmentation_probability:
            return self.validation_transforms(image=image, keypoints=keypoints)
        
        # Randomly select augmentation pipeline
        modes = ['light', 'medium', 'heavy', 'geometric', 'color']
        weights = [0.1, 0.4, 0.3, 0.15, 0.05]  # Prefer medium augmentations
        selected_mode = random.choices(modes, weights=weights)[0]
        
        transform = self.get_transforms(selected_mode)
        return transform(image=image, keypoints=keypoints)
    
    def create_progressive_augmentation(self, epoch: int, total_epochs: int) -> A.Compose:
        """
        Create progressive augmentation that increases in intensity over epochs.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Augmentation pipeline appropriate for current training stage
        """
        progress = epoch / total_epochs
        
        if progress < 0.2:
            return self.get_transforms('light')
        elif progress < 0.5:
            return self.get_transforms('medium')
        elif progress < 0.8:
            return self.get_transforms('heavy')
        else:
            # Final stage: mix of all augmentations
            return self.get_transforms('geometric')


class TTA_Augmentation:
    """Test Time Augmentation for inference."""
    
    def __init__(self, image_size: Tuple[int, int] = (480, 640)):
        """
        Initialize TTA augmentation.
        
        Args:
            image_size: Target image size (width, height)
        """
        self.image_size = image_size
        self.tta_transforms = self._create_tta_transforms()
    
    def _create_tta_transforms(self) -> List[A.Compose]:
        """Create list of TTA transforms."""
        base_transform = [
            A.Resize(height=self.image_size[1], width=self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        transforms = [
            # Original
            A.Compose(base_transform, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
            
            # Horizontal flip
            A.Compose([
                A.HorizontalFlip(p=1.0),
                *base_transform
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
            
            # Vertical flip
            A.Compose([
                A.VerticalFlip(p=1.0),
                *base_transform
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
            
            # Both flips
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                *base_transform
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
            
            # Rotate 90
            A.Compose([
                A.Rotate(limit=90, p=1.0),
                *base_transform
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
            
            # Rotate -90
            A.Compose([
                A.Rotate(limit=-90, p=1.0),
                *base_transform
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
        ]
        
        return transforms
    
    def apply_tta(
        self,
        image: np.ndarray,
        keypoints: List[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Apply all TTA transforms.
        
        Args:
            image: Input image
            keypoints: Original keypoints
            
        Returns:
            List of augmented samples
        """
        results = []
        for transform in self.tta_transforms:
            augmented = transform(image=image, keypoints=keypoints)
            results.append(augmented)
        
        return results


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create augmentation instance
    aug = CornerAugmentation(image_size=(480, 640))
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    dummy_keypoints = [(100, 150), (380, 150), (380, 490), (100, 490)]
    
    # Test different augmentation modes
    modes = ['light', 'medium', 'heavy', 'geometric', 'color']
    
    for mode in modes:
        print(f"Testing {mode} augmentation...")
        transform = aug.get_transforms(mode)
        result = transform(image=dummy_image, keypoints=dummy_keypoints)
        print(f"  - Image shape: {result['image'].shape}")
        print(f"  - Keypoints: {len(result['keypoints'])}")
    
    # Test TTA
    tta = TTA_Augmentation(image_size=(480, 640))
    tta_results = tta.apply_tta(dummy_image, dummy_keypoints)
    print(f"TTA generated {len(tta_results)} augmented samples")
    
    print("Augmentation module ready!")