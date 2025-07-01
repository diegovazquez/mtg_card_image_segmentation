#!/usr/bin/env python3
"""
Evaluation script for Magic: The Gathering card corner detection model.
Implements distance-based accuracy metrics (5px, 10px, 20px thresholds).
"""

import os
import sys
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml
from dataclasses import dataclass
import pandas as pd
from scipy.spatial.distance import euclidean


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    distance_accuracies: Dict[int, float]  # Accuracy at different pixel thresholds
    mean_distance_error: float
    median_distance_error: float
    std_distance_error: float
    corner_wise_accuracies: Dict[str, Dict[int, float]]  # Per-corner accuracies
    prediction_confidence: List[float]
    inference_times: List[float]
    total_samples: int
    successful_detections: int
    detection_rate: float
    distance_errors: List[List[float]]  # Raw distance errors for each sample


class CornerEvaluator:
    """Evaluator for corner detection model with distance-based metrics."""
    
    def __init__(self, image_size: Tuple[int, int] = (480, 640)):
        """
        Initialize evaluator.
        
        Args:
            image_size: Target image size (width, height)
        """
        self.image_size = image_size
        self.corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        
    def load_annotations(self, annotations_file: str, split: str = "test") -> Dict[str, List[List[float]]]:
        """
        Load ground truth annotations.
        
        Args:
            annotations_file: Path to corner_annotations.json
            split: Data split to load ('train' or 'test')
            
        Returns:
            Dictionary mapping image names to corner coordinates
        """
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        if split not in annotations:
            raise ValueError(f"Split '{split}' not found in annotations")
        
        return annotations[split]
    
    def predict_corners(self, model: YOLO, image_path: str) -> Tuple[Optional[List[Tuple[float, float]]], float, float]:
        """
        Predict corners for a single image.
        
        Args:
            model: Loaded YOLO model
            image_path: Path to input image
            
        Returns:
            Tuple of (predicted_corners, confidence, inference_time)
        """
        import time
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None, 0.0, 0.0
        
        original_height, original_width = image.shape[:2]
        
        # Run inference
        start_time = time.time()
        results = model.predict(
            source=image_path,
            imgsz=max(self.image_size),
            conf=0.1,  # Low confidence threshold to get all detections
            verbose=False,
            save=False
        )
        inference_time = time.time() - start_time
        
        if not results or len(results) == 0:
            return None, 0.0, inference_time
        
        result = results[0]
        
        # Extract keypoints and confidence
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None, 0.0, inference_time
        
        # Get the first (and should be only) detection
        keypoints = result.keypoints.data[0]  # Shape: [num_keypoints, 3] (x, y, confidence)
        confidence = float(result.boxes.conf[0]) if result.boxes is not None and len(result.boxes.conf) > 0 else 0.0
        
        # Extract corner coordinates
        corners = []
        for i in range(min(4, len(keypoints))):
            x, y, kp_conf = keypoints[i]
            
            # Convert from model coordinates to original image coordinates
            x = float(x) * original_width / self.image_size[0]
            y = float(y) * original_height / self.image_size[1]
            
            corners.append((x, y))
        
        # Ensure we have exactly 4 corners
        while len(corners) < 4:
            corners.append((0.0, 0.0))
        
        return corners[:4], confidence, inference_time
    
    def calculate_distance_error(
        self,
        predicted_corners: List[Tuple[float, float]],
        ground_truth_corners: List[List[float]]
    ) -> List[float]:
        """
        Calculate Euclidean distance errors between predicted and ground truth corners.
        
        Args:
            predicted_corners: List of predicted (x, y) coordinates
            ground_truth_corners: List of ground truth [x, y] coordinates
            
        Returns:
            List of distance errors for each corner
        """
        if len(predicted_corners) != 4 or len(ground_truth_corners) != 4:
            return [float('inf')] * 4
        
        errors = []
        for pred, gt in zip(predicted_corners, ground_truth_corners):
            error = euclidean(pred, gt[:2])
            errors.append(error)
        
        return errors
    
    def calculate_accuracy_at_threshold(
        self,
        distance_errors: List[List[float]],
        threshold: int
    ) -> float:
        """
        Calculate accuracy at a specific pixel distance threshold.
        
        Args:
            distance_errors: List of distance errors for all samples
            threshold: Distance threshold in pixels
            
        Returns:
            Accuracy as percentage of corners within threshold
        """
        total_corners = 0
        correct_corners = 0
        
        for sample_errors in distance_errors:
            for error in sample_errors:
                if not np.isinf(error):
                    total_corners += 1
                    if error <= threshold:
                        correct_corners += 1
        
        return correct_corners / total_corners if total_corners > 0 else 0.0
    
    def calculate_corner_wise_accuracy(
        self,
        distance_errors: List[List[float]],
        thresholds: List[int]
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate per-corner accuracy at different thresholds.
        
        Args:
            distance_errors: List of distance errors for all samples
            thresholds: List of distance thresholds
            
        Returns:
            Dictionary mapping corner names to threshold accuracies
        """
        corner_accuracies = {corner: {} for corner in self.corner_names}
        
        for threshold in thresholds:
            for corner_idx, corner_name in enumerate(self.corner_names):
                total = 0
                correct = 0
                
                for sample_errors in distance_errors:
                    if corner_idx < len(sample_errors) and not np.isinf(sample_errors[corner_idx]):
                        total += 1
                        if sample_errors[corner_idx] <= threshold:
                            correct += 1
                
                corner_accuracies[corner_name][threshold] = correct / total if total > 0 else 0.0
        
        return corner_accuracies
    
    def evaluate_model(
        self,
        model_path: str,
        annotations_file: str,
        images_dir: str,
        split: str = "test",
        distance_thresholds: List[int] = [5, 10, 20],
        max_samples: Optional[int] = None
    ) -> EvaluationMetrics:
        """
        Evaluate model performance with distance-based metrics.
        
        Args:
            model_path: Path to trained model weights
            annotations_file: Path to corner_annotations.json
            images_dir: Directory containing images
            split: Data split to evaluate ('test' or 'train')
            distance_thresholds: List of distance thresholds in pixels
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            EvaluationMetrics object with all metrics
        """
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path, task='pose')
        
        print(f"Loading annotations from: {annotations_file}")
        annotations = self.load_annotations(annotations_file, split)
        
        # Prepare evaluation data
        image_names = list(annotations.keys())
        if max_samples:
            image_names = image_names[:max_samples]
        
        print(f"Evaluating on {len(image_names)} images...")
        
        # Storage for results
        all_distance_errors = []
        all_confidences = []
        all_inference_times = []
        successful_detections = 0
        
        # Evaluate each image
        for image_name in tqdm(image_names, desc="Evaluating"):
            image_path = os.path.join(images_dir, split, "images", image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Get ground truth
            gt_corners = annotations[image_name]
            
            # Predict corners
            pred_corners, confidence, inference_time = self.predict_corners(model, image_path)
            
            all_inference_times.append(inference_time)
            
            if pred_corners is not None:
                # Calculate distance errors
                errors = self.calculate_distance_error(pred_corners, gt_corners)
                all_distance_errors.append(errors)
                all_confidences.append(confidence)
                successful_detections += 1
            else:
                # No detection - infinite error
                all_distance_errors.append([float('inf')] * 4)
                all_confidences.append(0.0)
        
        # Calculate metrics
        print("Calculating metrics...")
        
        # Overall accuracy at different thresholds
        distance_accuracies = {}
        for threshold in distance_thresholds:
            accuracy = self.calculate_accuracy_at_threshold(all_distance_errors, threshold)
            distance_accuracies[threshold] = accuracy
        
        # Per-corner accuracies
        corner_wise_accuracies = self.calculate_corner_wise_accuracy(all_distance_errors, distance_thresholds)
        
        # Distance error statistics
        all_valid_errors = []
        for sample_errors in all_distance_errors:
            for error in sample_errors:
                if not np.isinf(error):
                    all_valid_errors.append(error)
        
        mean_error = np.mean(all_valid_errors) if all_valid_errors else float('inf')
        median_error = np.median(all_valid_errors) if all_valid_errors else float('inf')
        std_error = np.std(all_valid_errors) if all_valid_errors else float('inf')
        
        # Detection rate
        detection_rate = successful_detections / len(image_names) if image_names else 0.0
        
        return EvaluationMetrics(
            distance_accuracies=distance_accuracies,
            mean_distance_error=mean_error,
            median_distance_error=median_error,
            std_distance_error=std_error,
            corner_wise_accuracies=corner_wise_accuracies,
            prediction_confidence=all_confidences,
            inference_times=all_inference_times,
            total_samples=len(image_names),
            successful_detections=successful_detections,
            detection_rate=detection_rate,
            distance_errors=all_distance_errors
        )
    
    def evaluate_model_from_data_config(
        self,
        model_path: str,
        data_config: str,
        distance_thresholds: List[int] = [5, 10, 20],
        max_samples: Optional[int] = None
    ) -> EvaluationMetrics:
        """
        Evaluate model using YOLO data configuration.
        
        Args:
            model_path: Path to trained model weights
            data_config: Path to data.yaml configuration
            distance_thresholds: List of distance thresholds in pixels
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            EvaluationMetrics object
        """
        # Load data configuration
        with open(data_config, 'r') as f:
            config = yaml.safe_load(f)
        
        val_dir = config['val']
        
        # Find images and labels
        images_dir = os.path.join(val_dir, 'images')
        labels_dir = os.path.join(val_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise ValueError(f"Images or labels directory not found in: {val_dir}")
        
        # Get image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if max_samples:
            image_files = image_files[:max_samples]
        
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        
        print(f"Evaluating on {len(image_files)} images...")
        
        # Storage for results
        all_distance_errors = []
        all_confidences = []
        all_inference_times = []
        successful_detections = 0
        
        # Evaluate each image
        for image_file in tqdm(image_files, desc="Evaluating"):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found: {label_path}")
                continue
            
            # Load ground truth from YOLO label file
            gt_corners = self._load_yolo_label(label_path, image_path)
            if gt_corners is None:
                continue
            
            # Predict corners
            pred_corners, confidence, inference_time = self.predict_corners(model, image_path)
            
            all_inference_times.append(inference_time)
            
            if pred_corners is not None:
                # Calculate distance errors
                errors = self.calculate_distance_error(pred_corners, gt_corners)
                all_distance_errors.append(errors)
                all_confidences.append(confidence)
                successful_detections += 1
            else:
                # No detection - infinite error
                all_distance_errors.append([float('inf')] * 4)
                all_confidences.append(0.0)
        
        # Calculate metrics (same as above)
        distance_accuracies = {}
        for threshold in distance_thresholds:
            accuracy = self.calculate_accuracy_at_threshold(all_distance_errors, threshold)
            distance_accuracies[threshold] = accuracy
        
        corner_wise_accuracies = self.calculate_corner_wise_accuracy(all_distance_errors, distance_thresholds)
        
        all_valid_errors = []
        for sample_errors in all_distance_errors:
            for error in sample_errors:
                if not np.isinf(error):
                    all_valid_errors.append(error)
        
        mean_error = np.mean(all_valid_errors) if all_valid_errors else float('inf')
        median_error = np.median(all_valid_errors) if all_valid_errors else float('inf')
        std_error = np.std(all_valid_errors) if all_valid_errors else float('inf')
        
        detection_rate = successful_detections / len(image_files) if image_files else 0.0
        
        return EvaluationMetrics(
            distance_accuracies=distance_accuracies,
            mean_distance_error=mean_error,
            median_distance_error=median_error,
            std_distance_error=std_error,
            corner_wise_accuracies=corner_wise_accuracies,
            prediction_confidence=all_confidences,
            inference_times=all_inference_times,
            total_samples=len(image_files),
            successful_detections=successful_detections,
            detection_rate=detection_rate,
            distance_errors=all_distance_errors
        )
    
    def _load_yolo_label(self, label_path: str, image_path: str) -> Optional[List[List[float]]]:
        """
        Load ground truth corners from YOLO label file.
        
        Args:
            label_path: Path to YOLO label file
            image_path: Path to corresponding image
            
        Returns:
            List of corner coordinates or None if failed
        """
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            height, width = image.shape[:2]
            
            # Read YOLO label
            with open(label_path, 'r') as f:
                line = f.readline().strip()
            
            if not line:
                return None
            
            parts = line.split()
            if len(parts) < 17:  # class + bbox (4) + keypoints (4*3)
                return None
            
            # Extract keypoints (skip class and bbox)
            keypoints = []
            for i in range(5, min(17, len(parts)), 3):  # Start after bbox, step by 3 (x, y, v)
                x_norm = float(parts[i])
                y_norm = float(parts[i + 1])
                
                # Convert normalized coordinates to image coordinates
                x = x_norm * width
                y = y_norm * height
                
                keypoints.append([x, y])
            
            return keypoints if len(keypoints) == 4 else None
            
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            return None
    
    def generate_evaluation_report(self, metrics: EvaluationMetrics, output_dir: str):
        """
        Generate comprehensive evaluation report with plots.
        
        Args:
            metrics: Evaluation metrics
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_dict = {
            'distance_accuracies': metrics.distance_accuracies,
            'mean_distance_error': metrics.mean_distance_error,
            'median_distance_error': metrics.median_distance_error,
            'std_distance_error': metrics.std_distance_error,
            'corner_wise_accuracies': metrics.corner_wise_accuracies,
            'total_samples': metrics.total_samples,
            'successful_detections': metrics.successful_detections,
            'detection_rate': metrics.detection_rate,
            'mean_inference_time': np.mean(metrics.inference_times),
            'mean_confidence': np.mean(metrics.prediction_confidence)
        }
        
        with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Generate plots
        self._plot_accuracy_by_threshold(metrics, output_dir)
        self._plot_corner_wise_accuracy(metrics, output_dir)
        self._plot_distance_error_distribution(metrics, output_dir)
        self._plot_confidence_distribution(metrics, output_dir)
        
        # Generate summary report
        self._generate_text_report(metrics, output_dir)
        
        print(f"Evaluation report saved to: {output_dir}")
    
    def _plot_accuracy_by_threshold(self, metrics: EvaluationMetrics, output_dir: str):
        """Plot accuracy vs distance threshold."""
        plt.figure(figsize=(10, 6))
        
        thresholds = sorted(metrics.distance_accuracies.keys())
        accuracies = [metrics.distance_accuracies[t] for t in thresholds]
        
        plt.plot(thresholds, accuracies, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Distance Threshold (pixels)')
        plt.ylabel('Accuracy')
        plt.title('Corner Detection Accuracy vs Distance Threshold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels
        for t, a in zip(thresholds, accuracies):
            plt.annotate(f'{a:.3f}', (t, a), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_threshold.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_corner_wise_accuracy(self, metrics: EvaluationMetrics, output_dir: str):
        """Plot per-corner accuracy comparison."""
        plt.figure(figsize=(12, 6))
        
        thresholds = sorted(list(metrics.corner_wise_accuracies[self.corner_names[0]].keys()))
        x = np.arange(len(thresholds))
        width = 0.2
        
        for i, corner in enumerate(self.corner_names):
            accuracies = [metrics.corner_wise_accuracies[corner][t] for t in thresholds]
            plt.bar(x + i * width, accuracies, width, label=corner.replace('_', ' ').title())
        
        plt.xlabel('Distance Threshold (pixels)')
        plt.ylabel('Accuracy')
        plt.title('Per-Corner Detection Accuracy')
        plt.xticks(x + width * 1.5, thresholds)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'corner_wise_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distance_error_distribution(self, metrics: EvaluationMetrics, output_dir: str):
        """Plot distribution of distance errors."""
        # Filter out infinite errors
        valid_errors = [error for error in np.concatenate([[error for error in sample_errors if not np.isinf(error) and not np.isnan(error)] 
                                                          for sample_errors in metrics.distance_errors if sample_errors]) if not np.isnan(error)]
        
        if not valid_errors:
            print("No valid distance errors to plot")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(valid_errors, bins=50, alpha=0.7, density=True)
        plt.xlabel('Distance Error (pixels)')
        plt.ylabel('Density')
        plt.title('Distance Error Distribution')
        plt.axvline(metrics.mean_distance_error, color='red', linestyle='--', label=f'Mean: {metrics.mean_distance_error:.2f}')
        plt.axvline(metrics.median_distance_error, color='green', linestyle='--', label=f'Median: {metrics.median_distance_error:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(valid_errors)
        plt.ylabel('Distance Error (pixels)')
        plt.title('Distance Error Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distance_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, metrics: EvaluationMetrics, output_dir: str):
        """Plot prediction confidence distribution."""
        plt.figure(figsize=(10, 6))
        
        plt.hist(metrics.prediction_confidence, bins=30, alpha=0.7, density=True)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution')
        plt.axvline(np.mean(metrics.prediction_confidence), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(metrics.prediction_confidence):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, metrics: EvaluationMetrics, output_dir: str):
        """Generate text summary report."""
        report_lines = [
            "CORNER DETECTION EVALUATION REPORT",
            "=" * 50,
            "",
            f"Total Samples: {metrics.total_samples}",
            f"Successful Detections: {metrics.successful_detections}",
            f"Detection Rate: {metrics.detection_rate:.3f}",
            "",
            "DISTANCE-BASED ACCURACY METRICS:",
            "-" * 35,
        ]
        
        for threshold, accuracy in sorted(metrics.distance_accuracies.items()):
            report_lines.append(f"Accuracy @ {threshold}px: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        report_lines.extend([
            "",
            "DISTANCE ERROR STATISTICS:",
            "-" * 27,
            f"Mean Distance Error: {metrics.mean_distance_error:.2f}px",
            f"Median Distance Error: {metrics.median_distance_error:.2f}px",
            f"Std Distance Error: {metrics.std_distance_error:.2f}px",
            "",
            "PER-CORNER ACCURACY:",
            "-" * 20,
        ])
        
        for corner in self.corner_names:
            report_lines.append(f"\n{corner.replace('_', ' ').title()}:")
            for threshold in sorted(metrics.corner_wise_accuracies[corner].keys()):
                accuracy = metrics.corner_wise_accuracies[corner][threshold]
                report_lines.append(f"  @ {threshold}px: {accuracy:.3f}")
        
        report_lines.extend([
            "",
            "PERFORMANCE STATISTICS:",
            "-" * 23,
            f"Mean Inference Time: {np.mean(metrics.inference_times):.4f}s",
            f"Mean Confidence: {np.mean(metrics.prediction_confidence):.3f}",
            ""
        ])
        
        # Save report
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate corner detection model")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str,
                       help='Path to data.yaml configuration (for YOLO format evaluation)')
    parser.add_argument('--annotations', type=str,
                       help='Path to corner_annotations.json (for direct evaluation)')
    parser.add_argument('--images-dir', type=str,
                       help='Directory containing images (for direct evaluation)')
    parser.add_argument('--split', type=str, default='test',
                       help='Data split to evaluate (test/train)')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[5, 10, 20],
                       help='Distance thresholds for accuracy calculation')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to evaluate')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Validate arguments
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if args.data and not os.path.exists(args.data):
        print(f"Error: Data config file not found: {args.data}")
        sys.exit(1)
    
    if args.annotations and not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)
    
    # Create evaluator
    evaluator = CornerEvaluator()
    
    print("Starting model evaluation...")
    print(f"Model: {args.model}")
    print(f"Thresholds: {args.thresholds}px")
    print(f"Output: {args.output_dir}")
    print("-" * 50)
    
    # Run evaluation
    if args.data:
        # YOLO format evaluation
        metrics = evaluator.evaluate_model_from_data_config(
            model_path=args.model,
            data_config=args.data,
            distance_thresholds=args.thresholds,
            max_samples=args.max_samples
        )
    elif args.annotations and args.images_dir:
        # Direct evaluation
        metrics = evaluator.evaluate_model(
            model_path=args.model,
            annotations_file=args.annotations,
            images_dir=args.images_dir,
            split=args.split,
            distance_thresholds=args.thresholds,
            max_samples=args.max_samples
        )
    else:
        print("Error: Either --data or both --annotations and --images-dir must be provided")
        sys.exit(1)
    
    # Generate report
    evaluator.generate_evaluation_report(metrics, args.output_dir)
    
    # Print summary
    print("\nEVALUATION RESULTS:")
    print(f"Detection Rate: {metrics.detection_rate:.3f}")
    print(f"Mean Distance Error: {metrics.mean_distance_error:.2f}px")
    
    for threshold, accuracy in sorted(metrics.distance_accuracies.items()):
        print(f"Accuracy @ {threshold}px: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()