"""
Utility functions for training, evaluation, and model management.
Includes loss functions, metrics, and training helpers.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class DiceLoss(nn.Module):
    """
    Dice loss for semantic segmentation.
    """
    
    def __init__(self, smooth=1e-6):
        """
        Initialize Dice loss.
        
        Args:
            smooth (float): Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Calculate Dice loss.
        
        Args:
            predictions (torch.Tensor): Model predictions (B, C, H, W)
            targets (torch.Tensor): Ground truth masks (B, H, W)
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Flatten tensors
        predictions = predictions.reshape(-1)
        targets_one_hot = targets_one_hot.reshape(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets_one_hot.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combined loss function using Dice loss and Cross-entropy loss.
    """
    
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None):
        """
        Initialize combined loss.
        
        Args:
            dice_weight (float): Weight for Dice loss
            ce_weight (float): Weight for Cross-entropy loss
            class_weights (torch.Tensor, optional): Class weights for CE loss
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss.
        
        Args:
            predictions (torch.Tensor): Model predictions (B, C, H, W)
            targets (torch.Tensor): Ground truth masks (B, H, W)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        dice_loss = self.dice_loss(predictions, targets)
        ce_loss = self.ce_loss(predictions, targets)
        
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

def calculate_iou(predictions, targets, num_classes=2, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for each class.
    
    Args:
        predictions (torch.Tensor): Model predictions (B, C, H, W)
        targets (torch.Tensor): Ground truth masks (B, H, W)
        num_classes (int): Number of classes
        smooth (float): Smoothing factor
        
    Returns:
        torch.Tensor: IoU values for each class
    """
    # Get predicted classes
    pred_classes = torch.argmax(predictions, dim=1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_classes == cls).float()
        target_cls = (targets == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)
    
    return torch.stack(ious)

def calculate_dice_coefficient(predictions, targets, num_classes=2, smooth=1e-6):
    """
    Calculate Dice coefficient for each class.
    
    Args:
        predictions (torch.Tensor): Model predictions (B, C, H, W)
        targets (torch.Tensor): Ground truth masks (B, H, W)
        num_classes (int): Number of classes
        smooth (float): Smoothing factor
        
    Returns:
        torch.Tensor: Dice coefficients for each class
    """
    # Get predicted classes
    pred_classes = torch.argmax(predictions, dim=1)
    
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred_classes == cls).float()
        target_cls = (targets == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        dice = (2. * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
        dice_scores.append(dice)
    
    return torch.stack(dice_scores)

def calculate_pixel_accuracy(predictions, targets):
    """
    Calculate pixel-wise accuracy.
    
    Args:
        predictions (torch.Tensor): Model predictions (B, C, H, W)
        targets (torch.Tensor): Ground truth masks (B, H, W)
        
    Returns:
        torch.Tensor: Pixel accuracy
    """
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy

class MetricsCalculator:
    """
    Class to calculate and accumulate metrics during training/validation.
    """
    
    def __init__(self, num_classes=2, device='cpu'):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes (int): Number of classes
            device (str or torch.device): Device to place tensors on
        """
        self.num_classes = num_classes
        self.device = torch.device(device) if isinstance(device, str) else device
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_loss = 0.0
        self.total_iou = torch.zeros(self.num_classes, device=self.device)
        self.total_dice = torch.zeros(self.num_classes, device=self.device)
        self.total_accuracy = 0.0
        self.count = 0
    
    def update(self, loss, predictions, targets):
        """
        Update metrics with batch results.
        
        Args:
            loss (torch.Tensor): Loss value for the batch
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth masks
        """
        self.total_loss += loss.item()
        self.total_iou += calculate_iou(predictions, targets, self.num_classes).to(self.device)
        self.total_dice += calculate_dice_coefficient(predictions, targets, self.num_classes).to(self.device)
        self.total_accuracy += calculate_pixel_accuracy(predictions, targets).item()
        self.count += 1
    
    def get_metrics(self):
        """
        Get averaged metrics.
        
        Returns:
            dict: Dictionary containing averaged metrics
        """
        if self.count == 0:
            return {}
        
        return {
            'loss': self.total_loss / self.count,
            'iou_background': self.total_iou[0].item() / self.count,
            'iou_card': self.total_iou[1].item() / self.count,
            'mean_iou': self.total_iou.mean().item() / self.count,
            'dice_background': self.total_dice[0].item() / self.count,
            'dice_card': self.total_dice[1].item() / self.count,
            'mean_dice': self.total_dice.mean().item() / self.count,
            'pixel_accuracy': self.total_accuracy / self.count
        }

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, checkpoint_dir, filename):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer state (can be None)
        scheduler: Learning rate scheduler state (can be None)
        epoch (int): Current epoch
        best_metric (float): Best validation metric achieved
        checkpoint_dir (str): Directory to save checkpoint
        filename (str): Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        tuple: (epoch, best_metric)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Resumed from epoch: {epoch}, Best metric: {best_metric:.4f}")
    
    return epoch, best_metric

def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """
    Plot training history.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_metrics (list): Training metrics
        val_metrics (list): Validation metrics
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot IoU
    train_iou = [m.get('mean_iou', 0) for m in train_metrics]
    val_iou = [m.get('mean_iou', 0) for m in val_metrics]
    axes[0, 1].plot(train_iou, label='Train IoU', color='blue')
    axes[0, 1].plot(val_iou, label='Val IoU', color='red')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot Dice
    train_dice = [m.get('mean_dice', 0) for m in train_metrics]
    val_dice = [m.get('mean_dice', 0) for m in val_metrics]
    axes[1, 0].plot(train_dice, label='Train Dice', color='blue')
    axes[1, 0].plot(val_dice, label='Val Dice', color='red')
    axes[1, 0].set_title('Mean Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot Pixel Accuracy
    train_acc = [m.get('pixel_accuracy', 0) for m in train_metrics]
    val_acc = [m.get('pixel_accuracy', 0) for m in val_metrics]
    axes[1, 1].plot(train_acc, label='Train Accuracy', color='blue')
    axes[1, 1].plot(val_acc, label='Val Accuracy', color='red')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
    
    plt.show()

def visualize_predictions(model, dataset, device, num_samples=4, save_path=None):
    """
    Visualize model predictions on samples from dataset.
    
    Args:
        model (nn.Module): Trained model
        dataset: Dataset to sample from
        device: Device to run inference on
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the visualization
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            true_mask = sample['mask'].numpy()
            
            # Get prediction
            prediction = model(image)
            pred_mask = torch.argmax(prediction, dim=1).cpu().numpy()[0]
            
            # Prepare image for visualization
            img_vis = sample['image'].permute(1, 2, 0).numpy()
            img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_vis = np.clip(img_vis, 0, 1)
            
            # Plot
            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title(f'Original Image - {sample["filename"]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth Mask')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved: {save_path}")
    
    plt.show()

def print_metrics(metrics, prefix=""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary of metrics
        prefix (str): Prefix for the output
    """
    print(f"{prefix}Metrics:")
    print(f"  Loss: {metrics.get('loss', 0):.4f}")
    print(f"  Mean IoU: {metrics.get('mean_iou', 0):.4f}")
    print(f"  IoU Background: {metrics.get('iou_background', 0):.4f}")
    print(f"  IoU Card: {metrics.get('iou_card', 0):.4f}")
    print(f"  Mean Dice: {metrics.get('mean_dice', 0):.4f}")
    print(f"  Dice Background: {metrics.get('dice_background', 0):.4f}")
    print(f"  Dice Card: {metrics.get('dice_card', 0):.4f}")
    print(f"  Pixel Accuracy: {metrics.get('pixel_accuracy', 0):.4f}")

if __name__ == "__main__":
    # Test loss functions
    batch_size, num_classes, height, width = 2, 2, 480, 640
    
    # Create dummy data
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test losses
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()
    
    dice_val = dice_loss(predictions, targets)
    combined_val = combined_loss(predictions, targets)
    
    print(f"Dice Loss: {dice_val:.4f}")
    print(f"Combined Loss: {combined_val:.4f}")
    
    # Test metrics
    iou = calculate_iou(predictions, targets)
    dice = calculate_dice_coefficient(predictions, targets)
    accuracy = calculate_pixel_accuracy(predictions, targets)
    
    print(f"IoU: {iou}")
    print(f"Dice: {dice}")
    print(f"Pixel Accuracy: {accuracy:.4f}")
