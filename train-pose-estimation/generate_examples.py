#!/usr/bin/env python3
"""
Generate training examples and visualizations for Magic: The Gathering card corner detection.
Creates visualization plots, data analysis, and sample predictions for model validation.
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
from ultralytics import YOLO


class ExampleGenerator:
    """Generate training examples and visualizations for corner detection."""
    
    def __init__(self, output_dir: str = "examples"):
        """
        Initialize example generator.
        
        Args:
            output_dir: Directory to save generated examples
        """
        self.output_dir = os.path.realpath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'annotations': os.path.join(self.output_dir, 'annotations'),
            'augmentations': os.path.join(self.output_dir, 'augmentations'),
            'predictions': os.path.join(self.output_dir, 'predictions'),
            'analysis': os.path.join(self.output_dir, 'analysis'),
            'samples': os.path.join(self.output_dir, 'samples')
        }
        
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        self.corner_colors = ['red', 'blue', 'green', 'orange']
    
    def load_annotations(self, annotations_file: str) -> Dict[str, Any]:
        """
        Load corner annotations from JSON file.
        
        Args:
            annotations_file: Path to corner_annotations.json
            
        Returns:
            Loaded annotations dictionary
        """
        with open(annotations_file, 'r') as f:
            return json.load(f)
    
    def visualize_annotations(
        self,
        annotations_file: str,
        images_dir: str,
        num_samples: int = 20,
        split: str = "train"
    ):
        """
        Visualize corner annotations on sample images.
        
        Args:
            annotations_file: Path to corner_annotations.json
            images_dir: Directory containing images
            num_samples: Number of samples to visualize
            split: Data split to use ('train' or 'test')
        """
        print(f"Generating annotation visualizations for {split} split...")
        
        annotations = self.load_annotations(annotations_file)
        
        if split not in annotations:
            print(f"Split '{split}' not found in annotations")
            return
        
        # Get random sample of images
        image_names = list(annotations[split].keys())
        sample_names = random.sample(image_names, min(num_samples, len(image_names)))
        
        # Create grid plot
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, image_name in enumerate(sample_names):
            if idx >= len(axes):
                break
            
            # Load image
            image_path = os.path.join(images_dir, split, "images", image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            corners = annotations[split][image_name]
            
            # Plot image with corners
            ax = axes[idx]
            ax.imshow(image)
            
            # Draw corners
            for i, (x, y) in enumerate(corners):
                ax.plot(x, y, 'o', color=self.corner_colors[i], 
                       markersize=8, markeredgewidth=2, markeredgecolor='white')
                ax.text(x + 10, y - 10, self.corner_names[i], 
                       color=self.corner_colors[i], fontweight='bold', fontsize=8)
            
            # Draw bounding box
            xs, ys = zip(*corners)
            bbox = patches.Rectangle(
                (min(xs), min(ys)), 
                max(xs) - min(xs), 
                max(ys) - min(ys),
                linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--'
            )
            ax.add_patch(bbox)
            
            # Draw card outline
            card_polygon = Polygon(corners, closed=True, fill=False, 
                                 edgecolor='cyan', linewidth=2, linestyle='-')
            ax.add_patch(card_polygon)
            
            ax.set_title(f"{image_name[:20]}...", fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(sample_names), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Corner Annotations - {split.capitalize()} Split', fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.subdirs['annotations'], f'{split}_annotations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Annotation visualization saved: {output_path}")
    
    def visualize_augmentations(
        self,
        annotations_file: str,
        images_dir: str,
        num_samples: int = 5,
        split: str = "train"
    ):
        """
        Visualize data augmentation effects on sample images.
        
        Args:
            annotations_file: Path to corner_annotations.json
            images_dir: Directory containing images
            num_samples: Number of samples to show
            split: Data split to use
        """
        print(f"Generating augmentation visualizations...")
        
        # Import augmentation module
        try:
            from augmentation import CornerAugmentation
        except ImportError:
            print("Warning: Could not import augmentation module")
            return
        
        annotations = self.load_annotations(annotations_file)
        
        if split not in annotations:
            print(f"Split '{split}' not found in annotations")
            return
        
        # Get sample images
        image_names = list(annotations[split].keys())
        sample_names = random.sample(image_names, min(num_samples, len(image_names)))
        
        # Initialize augmentation
        aug = CornerAugmentation(image_size=(480, 640))
        aug_modes = ['light', 'medium', 'heavy', 'geometric', 'color']
        
        for image_name in sample_names:
            # Load image and corners
            image_path = os.path.join(images_dir, split, "images", image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            corners = annotations[split][image_name]
            keypoints = [(float(x), float(y)) for x, y in corners]
            
            # Create subplot for different augmentations
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Original image
            axes[0].imshow(image)
            for i, (x, y) in enumerate(keypoints):
                axes[0].plot(x, y, 'o', color=self.corner_colors[i], markersize=6)
            axes[0].set_title('Original', fontsize=12)
            axes[0].axis('off')
            
            # Augmented versions
            for idx, mode in enumerate(aug_modes):
                try:
                    transform = aug.get_transforms(mode)
                    result = transform(image=image, keypoints=keypoints)
                    
                    aug_image = result['image']
                    aug_keypoints = result['keypoints']
                    
                    # Convert tensor to numpy if needed
                    if hasattr(aug_image, 'numpy'):
                        aug_image = aug_image.numpy()
                        aug_image = np.transpose(aug_image, (1, 2, 0))
                        # Denormalize
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        aug_image = aug_image * std + mean
                        aug_image = np.clip(aug_image, 0, 1)
                    
                    axes[idx + 1].imshow(aug_image)
                    for i, (x, y) in enumerate(aug_keypoints):
                        axes[idx + 1].plot(x, y, 'o', color=self.corner_colors[i], markersize=6)
                    axes[idx + 1].set_title(f'{mode.capitalize()} Augmentation', fontsize=12)
                    axes[idx + 1].axis('off')
                    
                except Exception as e:
                    print(f"Warning: Failed to apply {mode} augmentation: {e}")
                    axes[idx + 1].text(0.5, 0.5, f'Failed: {mode}', 
                                      ha='center', va='center', transform=axes[idx + 1].transAxes)
                    axes[idx + 1].axis('off')
            
            plt.suptitle(f'Augmentation Examples - {image_name[:30]}...', fontsize=14)
            plt.tight_layout()
            
            output_path = os.path.join(self.subdirs['augmentations'], 
                                     f'augmentation_{image_name.replace(".jpg", ".png")}')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Augmentation visualizations saved to: {self.subdirs['augmentations']}")
    
    def generate_dataset_analysis(self, annotations_file: str):
        """
        Generate comprehensive dataset analysis and statistics.
        
        Args:
            annotations_file: Path to corner_annotations.json
        """
        print("Generating dataset analysis...")
        
        annotations = self.load_annotations(annotations_file)
        
        # Collect statistics
        stats = {
            'splits': {},
            'corner_distributions': {'x': [], 'y': [], 'corner_id': []},
            'bbox_sizes': {'width': [], 'height': [], 'area': []},
            'card_types': {'full_art': 0, 'normal': 0, 'other': 0}
        }
        
        for split_name, split_data in annotations.items():
            stats['splits'][split_name] = {
                'count': len(split_data),
                'images': list(split_data.keys())
            }
            
            for image_name, corners in split_data.items():
                # Card type analysis
                if 'full_art' in image_name:
                    stats['card_types']['full_art'] += 1
                elif 'normal' in image_name:
                    stats['card_types']['normal'] += 1
                else:
                    stats['card_types']['other'] += 1
                
                # Corner distribution analysis
                for i, (x, y) in enumerate(corners):
                    stats['corner_distributions']['x'].append(x)
                    stats['corner_distributions']['y'].append(y)
                    stats['corner_distributions']['corner_id'].append(i)
                
                # Bounding box analysis
                xs, ys = zip(*corners)
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                area = width * height
                
                stats['bbox_sizes']['width'].append(width)
                stats['bbox_sizes']['height'].append(height)
                stats['bbox_sizes']['area'].append(area)
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Split distribution
        splits = list(stats['splits'].keys())
        counts = [stats['splits'][split]['count'] for split in splits]
        axes[0, 0].pie(counts, labels=splits, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Dataset Split Distribution')
        
        # Card type distribution
        card_types = list(stats['card_types'].keys())
        card_counts = list(stats['card_types'].values())
        axes[0, 1].bar(card_types, card_counts, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Card Type Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # Corner distribution heatmap
        x_coords = stats['corner_distributions']['x']
        y_coords = stats['corner_distributions']['y']
        axes[0, 2].hist2d(x_coords, y_coords, bins=50, cmap='viridis')
        axes[0, 2].set_title('Corner Position Heatmap')
        axes[0, 2].set_xlabel('X Coordinate')
        axes[0, 2].set_ylabel('Y Coordinate')
        
        # Bounding box width distribution
        axes[1, 0].hist(stats['bbox_sizes']['width'], bins=30, alpha=0.7, color='blue')
        axes[1, 0].set_title('Bounding Box Width Distribution')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Bounding box height distribution
        axes[1, 1].hist(stats['bbox_sizes']['height'], bins=30, alpha=0.7, color='green')
        axes[1, 1].set_title('Bounding Box Height Distribution')
        axes[1, 1].set_xlabel('Height (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        
        # Bounding box area distribution
        axes[1, 2].hist(stats['bbox_sizes']['area'], bins=30, alpha=0.7, color='red')
        axes[1, 2].set_title('Bounding Box Area Distribution')
        axes[1, 2].set_xlabel('Area (pixels²)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save analysis plot
        analysis_path = os.path.join(self.subdirs['analysis'], 'dataset_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to JSON
        stats_path = os.path.join(self.subdirs['analysis'], 'dataset_statistics.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {
            'splits': stats['splits'],
            'card_types': stats['card_types'],
            'corner_statistics': {
                'x_mean': float(np.mean(x_coords)),
                'x_std': float(np.std(x_coords)),
                'y_mean': float(np.mean(y_coords)),
                'y_std': float(np.std(y_coords))
            },
            'bbox_statistics': {
                'width_mean': float(np.mean(stats['bbox_sizes']['width'])),
                'width_std': float(np.std(stats['bbox_sizes']['width'])),
                'height_mean': float(np.mean(stats['bbox_sizes']['height'])),
                'height_std': float(np.std(stats['bbox_sizes']['height'])),
                'area_mean': float(np.mean(stats['bbox_sizes']['area'])),
                'area_std': float(np.std(stats['bbox_sizes']['area']))
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"Dataset analysis saved to: {self.subdirs['analysis']}")
        print(f"  - Analysis plot: {analysis_path}")
        print(f"  - Statistics JSON: {stats_path}")
    
    def visualize_model_predictions(
        self,
        model_path: str,
        annotations_file: str,
        images_dir: str,
        num_samples: int = 10,
        split: str = "test"
    ):
        """
        Visualize model predictions compared to ground truth.
        
        Args:
            model_path: Path to trained model
            annotations_file: Path to corner_annotations.json
            images_dir: Directory containing images
            num_samples: Number of samples to visualize
            split: Data split to use
        """
        print(f"Generating prediction visualizations...")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
        
        # Load model
        model = YOLO(model_path)
        annotations = self.load_annotations(annotations_file)
        
        if split not in annotations:
            print(f"Split '{split}' not found in annotations")
            return
        
        # Get sample images
        image_names = list(annotations[split].keys())
        sample_names = random.sample(image_names, min(num_samples, len(image_names)))
        
        # Create grid plot
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, image_name in enumerate(sample_names):
            if idx >= len(axes):
                break
            
            # Load image
            image_path = os.path.join(images_dir, split, "images", image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt_corners = annotations[split][image_name]
            
            # Run prediction
            try:
                results = model.predict(image_path, verbose=False, save=False)
                
                if results and len(results) > 0 and results[0].keypoints is not None:
                    # Extract predicted keypoints
                    keypoints = results[0].keypoints.data[0]  # First detection
                    height, width = image.shape[:2]
                    
                    pred_corners = []
                    for i in range(min(4, len(keypoints))):
                        x, y, conf = keypoints[i]
                        # Convert to image coordinates if normalized
                        if x <= 1.0 and y <= 1.0:
                            x *= width
                            y *= height
                        pred_corners.append((float(x), float(y)))
                else:
                    pred_corners = [(0, 0)] * 4  # No detection
            
            except Exception as e:
                print(f"Prediction failed for {image_name}: {e}")
                pred_corners = [(0, 0)] * 4
            
            # Plot image with both ground truth and predictions
            ax = axes[idx]
            ax.imshow(image_rgb)
            
            # Draw ground truth corners (circles)
            for i, (x, y) in enumerate(gt_corners):
                ax.plot(x, y, 'o', color=self.corner_colors[i], 
                       markersize=10, markeredgewidth=2, markeredgecolor='white',
                       label=f'GT {self.corner_names[i]}' if idx == 0 else '')
            
            # Draw predicted corners (squares)
            for i, (x, y) in enumerate(pred_corners):
                if x > 0 and y > 0:  # Valid prediction
                    ax.plot(x, y, 's', color=self.corner_colors[i], 
                           markersize=8, markeredgewidth=2, markeredgecolor='black',
                           alpha=0.7, label=f'Pred {self.corner_names[i]}' if idx == 0 else '')
            
            # Draw connection lines between GT and predictions
            for i in range(min(len(gt_corners), len(pred_corners))):
                if pred_corners[i][0] > 0 and pred_corners[i][1] > 0:
                    ax.plot([gt_corners[i][0], pred_corners[i][0]], 
                           [gt_corners[i][1], pred_corners[i][1]], 
                           '--', color='gray', alpha=0.5, linewidth=1)
            
            ax.set_title(f"{image_name[:20]}...", fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(sample_names), len(axes)):
            axes[idx].axis('off')
        
        # Add legend to first subplot
        if len(axes) > 0:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.suptitle(f'Model Predictions vs Ground Truth - {split.capitalize()} Split', fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.subdirs['predictions'], f'{split}_predictions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction visualization saved: {output_path}")
    
    def generate_sample_images(
        self,
        annotations_file: str,
        images_dir: str,
        num_samples: int = 50
    ):
        """
        Generate a collection of sample images with annotations.
        
        Args:
            annotations_file: Path to corner_annotations.json
            images_dir: Directory containing images
            num_samples: Number of sample images to generate
        """
        print(f"Generating sample image collection...")
        
        annotations = self.load_annotations(annotations_file)
        
        # Collect all images from all splits
        all_images = []
        for split_name, split_data in annotations.items():
            for image_name, corners in split_data.items():
                all_images.append({
                    'name': image_name,
                    'split': split_name,
                    'corners': corners,
                    'path': os.path.join(images_dir, split_name, "images", image_name)
                })
        
        # Sample random images
        sample_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        # Copy sample images with annotation overlay
        for i, img_info in enumerate(tqdm(sample_images, desc="Generating samples")):
            image = cv2.imread(img_info['path'])
            if image is None:
                continue
            
            # Draw corners and annotations
            for j, (x, y) in enumerate(img_info['corners']):
                # Draw corner point
                cv2.circle(image, (int(x), int(y)), 8, 
                          self._get_bgr_color(self.corner_colors[j]), -1)
                cv2.circle(image, (int(x), int(y)), 10, (255, 255, 255), 2)
                
                # Add corner label
                cv2.putText(image, f"C{j+1}", (int(x)+15, int(y)-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           self._get_bgr_color(self.corner_colors[j]), 2)
            
            # Draw bounding box
            corners = img_info['corners']
            xs, ys = zip(*corners)
            bbox_pts = np.array([
                [min(xs), min(ys)],
                [max(xs), min(ys)],
                [max(xs), max(ys)],
                [min(xs), max(ys)]
            ], dtype=np.int32)
            cv2.rectangle(image, tuple(bbox_pts[0]), tuple(bbox_pts[2]), (0, 255, 255), 2)
            
            # Draw card outline
            card_pts = np.array(corners, dtype=np.int32)
            cv2.polylines(image, [card_pts], True, (0, 255, 0), 3)
            
            # Add image info
            info_text = f"{img_info['split'].upper()} | {img_info['name'][:30]}"
            cv2.putText(image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            # Save annotated image
            output_name = f"sample_{i+1:03d}_{img_info['name']}"
            output_path = os.path.join(self.subdirs['samples'], output_name)
            cv2.imwrite(output_path, image)
        
        print(f"Sample images saved to: {self.subdirs['samples']}")
    
    def _get_bgr_color(self, color_name: str) -> Tuple[int, int, int]:
        """Convert color name to BGR tuple for OpenCV."""
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        return color_map.get(color_name, (255, 255, 255))
    
    def generate_all_examples(
        self,
        annotations_file: str,
        images_dir: str,
        model_path: Optional[str] = None,
        num_samples: int = 20
    ):
        """
        Generate all types of examples and visualizations.
        
        Args:
            annotations_file: Path to corner_annotations.json
            images_dir: Directory containing images
            model_path: Path to trained model (optional)
            num_samples: Number of samples for visualizations
        """
        print("Generating comprehensive example collection...")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        # Generate dataset analysis
        self.generate_dataset_analysis(annotations_file)
        
        # Generate annotation visualizations
        for split in ['train', 'test']:
            self.visualize_annotations(annotations_file, images_dir, num_samples, split)
        
        # Generate augmentation examples
        self.visualize_augmentations(annotations_file, images_dir, 5, 'train')
        
        # Generate model predictions if model is available
        if model_path and os.path.exists(model_path):
            self.visualize_model_predictions(model_path, annotations_file, images_dir, num_samples, 'test')
        
        # Generate sample image collection
        self.generate_sample_images(annotations_file, images_dir, num_samples * 2)
        
        # Create summary report
        self._create_summary_report(annotations_file, model_path)
        
        print("-" * 50)
        print(f"Example generation completed!")
        print(f"All outputs saved to: {self.output_dir}")
    
    def _create_summary_report(self, annotations_file: str, model_path: Optional[str]):
        """Create a summary report of generated examples."""
        
        report_lines = [
            "CORNER DETECTION TRAINING EXAMPLES REPORT",
            "=" * 50,
            "",
            f"Generated: {os.path.basename(self.output_dir)}",
            f"Annotations file: {annotations_file}",
            f"Model file: {model_path if model_path else 'Not provided'}",
            "",
            "GENERATED CONTENT:",
            "-" * 20,
            "",
            "📁 annotations/",
            "  └── Visualizations of corner annotations on sample images",
            "  └── Shows ground truth corner positions for train/test splits",
            "",
            "📁 augmentations/", 
            "  └── Examples of data augmentation effects",
            "  └── Shows how augmentations transform images and keypoints",
            "",
            "📁 analysis/",
            "  └── Comprehensive dataset statistics and analysis",
            "  └── Distribution plots, corner heatmaps, bbox statistics",
            "",
            "📁 predictions/",
            "  └── Model predictions vs ground truth comparisons",
            "  └── Visual validation of model performance",
            "",
            "📁 samples/",
            "  └── Collection of annotated sample images",
            "  └── Individual images with corner annotations overlay",
            "",
            "USAGE:",
            "-" * 10,
            "• Use annotation visualizations to verify data quality",
            "• Review augmentation examples to tune augmentation settings", 
            "• Analyze dataset statistics to understand data distribution",
            "• Compare predictions to evaluate model performance",
            "• Use sample images for presentations or documentation",
            "",
            "FILES ORGANIZATION:",
            "-" * 20,
        ]
        
        # Add file listing
        for subdir_name, subdir_path in self.subdirs.items():
            files = os.listdir(subdir_path) if os.path.exists(subdir_path) else []
            report_lines.append(f"• {subdir_name}/ ({len(files)} files)")
            for file in sorted(files)[:5]:  # Show first 5 files
                report_lines.append(f"  - {file}")
            if len(files) > 5:
                report_lines.append(f"  ... and {len(files) - 5} more files")
            report_lines.append("")
        
        # Save report
        report_path = os.path.join(self.output_dir, "README.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved: {report_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate training examples for corner detection")
    
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to corner_annotations.json')
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory containing train/test images')
    parser.add_argument('--model', type=str,
                       help='Path to trained model (optional)')
    parser.add_argument('--output-dir', type=str, default='examples',
                       help='Output directory for examples')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples for visualizations')
    parser.add_argument('--type', choices=['all', 'annotations', 'augmentations', 'analysis', 'predictions', 'samples'],
                       default='all', help='Type of examples to generate')
    
    return parser.parse_args()


def main():
    """Main example generation function."""
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)
    
    # Create generator
    generator = ExampleGenerator(args.output_dir)
    
    print("Magic: The Gathering Corner Detection - Example Generator")
    print(f"Annotations: {args.annotations}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Type: {args.type}")
    print("-" * 60)
    
    # Generate examples based on type
    if args.type == 'all':
        generator.generate_all_examples(
            annotations_file=args.annotations,
            images_dir=args.images_dir,
            model_path=args.model,
            num_samples=args.num_samples
        )
    elif args.type == 'annotations':
        for split in ['train', 'test']:
            generator.visualize_annotations(args.annotations, args.images_dir, args.num_samples, split)
    elif args.type == 'augmentations':
        generator.visualize_augmentations(args.annotations, args.images_dir, 5, 'train')
    elif args.type == 'analysis':
        generator.generate_dataset_analysis(args.annotations)
    elif args.type == 'predictions':
        if not args.model:
            print("Error: Model path required for prediction visualization")
            sys.exit(1)
        generator.visualize_model_predictions(args.model, args.annotations, args.images_dir, args.num_samples, 'test')
    elif args.type == 'samples':
        generator.generate_sample_images(args.annotations, args.images_dir, args.num_samples * 2)


if __name__ == "__main__":
    main()