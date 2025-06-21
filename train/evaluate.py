"""
Model evaluation script for semantic segmentation.
Provides comprehensive evaluation metrics and visualization capabilities.
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import Config
from model import create_model
from dataset import create_dataloaders, CardSegmentationDataset, get_validation_transforms
from utils import (
    CombinedLoss, MetricsCalculator, calculate_iou, calculate_dice_coefficient,
    calculate_pixel_accuracy, visualize_predictions, print_metrics
)

class ModelEvaluator:
    """
    Comprehensive model evaluator for semantic segmentation.
    """
    
    def __init__(self, model, device, num_classes=2):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): Model to evaluate
            device: Device to run evaluation on
            num_classes (int): Number of classes
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.eval()
    
    def evaluate_dataset(self, dataloader, criterion=None):
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            criterion: Loss function (optional)
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        print(f"Evaluating on {len(dataloader.dataset)} samples...")
        
        metrics_calc = MetricsCalculator(num_classes=self.num_classes, device=self.device)
        all_predictions = []
        all_targets = []
        all_filenames = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                filenames = batch['filename']
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss if criterion provided
                if criterion:
                    loss = criterion(outputs, masks)
                    metrics_calc.update(loss, outputs, masks)
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                
                # Store for confusion matrix
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
                all_filenames.extend(filenames)
                
                if batch_idx % 20 == 0:
                    print(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Get basic metrics
        basic_metrics = metrics_calc.get_metrics()
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=range(self.num_classes))
        
        # Calculate per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(cm)
        
        return {
            'basic_metrics': basic_metrics,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'filenames': all_filenames
        }
    
    def _calculate_per_class_metrics(self, cm):
        """
        Calculate per-class metrics from confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            
        Returns:
            dict: Per-class metrics
        """
        metrics = {}
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # IoU
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            class_name = 'background' if i == 0 else 'card'
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou,
                'support': cm[i, :].sum()
            }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        class_names = ['Background', 'Card']
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        
        plt.show()
    
    def analyze_predictions(self, dataset, num_samples=8, save_path=None):
        """
        Analyze model predictions on sample images.
        
        Args:
            dataset: Dataset to sample from
            num_samples (int): Number of samples to analyze
            save_path (str, optional): Path to save visualization
        """
        # Select samples (including some diverse cases)
        indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                sample = dataset[idx]
                image = sample['image'].unsqueeze(0).to(self.device)
                true_mask = sample['mask'].numpy()
                filename = sample['filename']
                
                # Get prediction
                output = self.model(image)
                pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                # Get prediction confidence
                prob_map = F.softmax(output, dim=1)[0, 1].cpu().numpy()  # Card probability
                
                # Prepare image for visualization
                img_vis = sample['image'].permute(1, 2, 0).numpy()
                img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_vis = np.clip(img_vis, 0, 1)
                
                # Calculate metrics for this sample
                sample_iou = calculate_iou(
                    output, torch.tensor(true_mask).unsqueeze(0).unsqueeze(0).to(self.device), 
                    num_classes=2
                )
                sample_dice = calculate_dice_coefficient(
                    output, torch.tensor(true_mask).unsqueeze(0).unsqueeze(0).to(self.device),
                    num_classes=2
                )
                
                # Plot
                axes[i, 0].imshow(img_vis)
                axes[i, 0].set_title(f'Image\n{filename}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                axes[i, 2].set_title(f'Prediction\nIoU: {sample_iou[1]:.3f}')
                axes[i, 2].axis('off')
                
                axes[i, 3].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
                axes[i, 3].set_title(f'Confidence\nDice: {sample_dice[1]:.3f}')
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction analysis saved: {save_path}")
        
        plt.show()
    
    def find_failure_cases(self, dataset, threshold_iou=0.5, num_cases=5):
        """
        Find and analyze failure cases.
        
        Args:
            dataset: Dataset to analyze
            threshold_iou (float): IoU threshold for failure
            num_cases (int): Number of failure cases to show
            
        Returns:
            list: List of failure case information
        """
        print(f"Finding failure cases with IoU < {threshold_iou}...")
        
        failure_cases = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                sample = dataset[idx]
                image = sample['image'].unsqueeze(0).to(self.device)
                true_mask = sample['mask'].numpy()
                filename = sample['filename']
                
                # Get prediction
                output = self.model(image)
                pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                # Calculate IoU for card class
                sample_iou = calculate_iou(
                    output, torch.tensor(true_mask).unsqueeze(0).unsqueeze(0).to(self.device),
                    num_classes=2
                )
                card_iou = sample_iou[1].item()
                
                if card_iou < threshold_iou:
                    failure_cases.append({
                        'idx': idx,
                        'filename': filename,
                        'iou': card_iou,
                        'true_mask': true_mask,
                        'pred_mask': pred_mask
                    })
                
                if idx % 100 == 0:
                    print(f"Analyzed {idx}/{len(dataset)} samples")
        
        # Sort by IoU (worst first)
        failure_cases.sort(key=lambda x: x['iou'])
        
        print(f"Found {len(failure_cases)} failure cases")
        
        # Show worst cases
        if failure_cases:
            self._visualize_failure_cases(dataset, failure_cases[:num_cases])
        
        return failure_cases
    
    def _visualize_failure_cases(self, dataset, failure_cases):
        """Visualize failure cases."""
        if not failure_cases:
            return
        
        num_cases = len(failure_cases)
        fig, axes = plt.subplots(num_cases, 3, figsize=(12, 4*num_cases))
        if num_cases == 1:
            axes = axes.reshape(1, -1)
        
        for i, case in enumerate(failure_cases):
            sample = dataset[case['idx']]
            
            # Prepare image for visualization
            img_vis = sample['image'].permute(1, 2, 0).numpy()
            img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_vis = np.clip(img_vis, 0, 1)
            
            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title(f'Image\n{case["filename"]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(case['true_mask'], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(case['pred_mask'], cmap='gray')
            axes[i, 2].set_title(f'Prediction\nIoU: {case["iou"]:.3f}')
            axes[i, 2].axis('off')
        
        plt.suptitle('Failure Cases (Worst IoU)', fontsize=16)
        plt.tight_layout()
        plt.show()

def main(args):
    """Main evaluation function."""
    
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Set device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = create_model(
        num_classes=Config.NUM_CLASSES,
        pretrained=False  # We're loading trained weights
    ).to(device)
    
    # Load model weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from: {args.model_path}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, Config.NUM_CLASSES)
    
    # Create test dataset
    test_dataset = CardSegmentationDataset(
        image_dir=Config.TEST_IMAGE_DIR,
        mask_dir=Config.TEST_MASK_DIR,
        transform=get_validation_transforms((Config.INPUT_HEIGHT, Config.INPUT_WIDTH))
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("No test data found. Please check dataset paths.")
        return
    
    # Create data loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Create loss function
    criterion = CombinedLoss(
        dice_weight=Config.DICE_WEIGHT,
        ce_weight=Config.BCE_WEIGHT
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluator.evaluate_dataset(test_loader, criterion)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    # Basic metrics
    print_metrics(results['basic_metrics'], "Test ")
    
    # Per-class metrics
    print("\nPer-class Metrics:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"\n{class_name.title()}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Support: {metrics['support']:,}")
    
    # Plot confusion matrix
    if args.save_plots:
        os.makedirs('train/evaluation', exist_ok=True)
        cm_path = 'train/evaluation/confusion_matrix.png'
        evaluator.plot_confusion_matrix(results['confusion_matrix'], cm_path)
    else:
        evaluator.plot_confusion_matrix(results['confusion_matrix'])
    
    # Analyze predictions
    if args.analyze_predictions:
        print("\nAnalyzing predictions...")
        if args.save_plots:
            pred_path = 'train/evaluation/prediction_analysis.png'
            evaluator.analyze_predictions(test_dataset, num_samples=8, save_path=pred_path)
        else:
            evaluator.analyze_predictions(test_dataset, num_samples=8)
    
    # Find failure cases
    if args.find_failures:
        print("\nFinding failure cases...")
        failure_cases = evaluator.find_failure_cases(
            test_dataset, 
            threshold_iou=args.failure_threshold,
            num_cases=5
        )
        
        if failure_cases:
            print(f"\nWorst {min(5, len(failure_cases))} failure cases:")
            for i, case in enumerate(failure_cases[:5]):
                print(f"{i+1}. {case['filename']}: IoU = {case['iou']:.3f}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate semantic segmentation model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--analyze-predictions', action='store_true',
                        help='Analyze and visualize predictions')
    parser.add_argument('--find-failures', action='store_true',
                        help='Find and analyze failure cases')
    parser.add_argument('--failure-threshold', type=float, default=0.5,
                        help='IoU threshold for failure cases (default: 0.5)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        raise
