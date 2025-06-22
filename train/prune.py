"""
Model pruning script for semantic segmentation.
Implements structured and unstructured pruning with fine-tuning.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import copy

from config import Config
from model import create_model, count_parameters, get_model_size
from dataset import create_dataloaders
from utils import (
    CombinedLoss, MetricsCalculator, save_checkpoint, load_checkpoint,
    print_metrics
)
from train import train_epoch, validate_epoch, create_optimizer, create_scheduler

class ModelPruner:
    """
    Class for pruning neural network models with different strategies.
    """
    
    def __init__(self, model, pruning_amount=0.3, structured=False):
        """
        Initialize model pruner.
        
        Args:
            model (nn.Module): Model to prune
            pruning_amount (float): Fraction of parameters to prune
            structured (bool): Use structured pruning
        """
        self.model = model
        self.pruning_amount = pruning_amount
        self.structured = structured
        self.original_accuracy = None
        self.pruned_accuracy = None
    
    def get_prunable_modules(self):
        """
        Get list of modules that can be pruned.
        
        Returns:
            list: List of (module, name) tuples for prunable parameters
        """
        prunable_modules = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prunable_modules.append((module, 'weight'))
            elif isinstance(module, nn.Linear):
                prunable_modules.append((module, 'weight'))
        
        return prunable_modules
    
    def apply_unstructured_pruning(self):
        """Apply unstructured (magnitude-based) pruning."""
        print(f"Applying unstructured pruning with {self.pruning_amount:.1%} sparsity...")
        
        # Get prunable modules
        prunable_modules = self.get_prunable_modules()
        
        # Apply magnitude-based pruning
        prune.global_unstructured(
            prunable_modules,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_amount,
        )
        
        print(f"Applied unstructured pruning to {len(prunable_modules)} modules")
    
    def apply_structured_pruning(self):
        """Apply structured pruning (remove entire channels/filters)."""
        print(f"Applying structured pruning with {self.pruning_amount:.1%} sparsity...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                # Prune entire output channels
                n_channels_to_prune = int(module.out_channels * self.pruning_amount)
                if n_channels_to_prune > 0:
                    prune.ln_structured(
                        module, 
                        name='weight', 
                        amount=n_channels_to_prune, 
                        n=2, 
                        dim=0
                    )
        
        print("Applied structured pruning to convolutional layers")
    
    def prune_model(self):
        """Apply pruning to the model."""
        if self.structured:
            self.apply_structured_pruning()
        else:
            self.apply_unstructured_pruning()
    
    def remove_pruning_masks(self):
        """
        Remove pruning masks and make pruning permanent.
        This reduces memory usage and computation.
        """
        print("Removing pruning masks...")
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')
    
    def get_sparsity_info(self):
        """
        Get sparsity information for the model.
        
        Returns:
            dict: Dictionary containing sparsity statistics
        """
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                if hasattr(module, 'weight_mask'):
                    weight = weight * module.weight_mask
                
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'sparsity': sparsity,
            'compression_ratio': 1 / (1 - sparsity) if sparsity < 1 else float('inf')
        }

def evaluate_model(model, val_loader, criterion, device):
    """
    Evaluate model performance.
    
    Args:
        model (nn.Module): Model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator(num_classes=Config.NUM_CLASSES, device=device)
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            with autocast('cuda', enabled=Config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            metrics_calc.update(loss, outputs, masks)
    
    return metrics_calc.get_metrics()

def fine_tune_pruned_model(model, train_loader, val_loader, device, epochs=20):
    """
    Fine-tune pruned model to recover performance.
    
    Args:
        model (nn.Module): Pruned model to fine-tune
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs (int): Number of fine-tuning epochs
        
    Returns:
        nn.Module: Fine-tuned model
    """
    print(f"Fine-tuning pruned model for {epochs} epochs...")
    
    # Create optimizer with lower learning rate
    optimizer = create_optimizer(model, Config)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1  # Reduce learning rate for fine-tuning
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, Config, train_loader)
    
    # Create loss function
    criterion = CombinedLoss(
        dice_weight=Config.DICE_WEIGHT,
        ce_weight=Config.BCE_WEIGHT
    )
    
    # Create gradient scaler
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    best_metric = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validation
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Save best model
        current_metric = val_metrics.get('mean_iou', 0)
        if current_metric > best_metric:
            best_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
        
        print(f"Fine-tune Epoch {epoch}/{epochs-1}")
        print_metrics(train_metrics, "Train ")
        print_metrics(val_metrics, "Val ")
        print("-" * 30)
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"Fine-tuning completed. Best validation IoU: {best_metric:.4f}")
    return model

def main(args):
    """Main pruning function."""
    
    print("=" * 50)
    print("MODEL PRUNING")
    print("=" * 50)
    
    # Create directories
    Config.create_directories()
    
    # Set device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_image_dir=Config.TRAIN_IMAGE_DIR,
        train_mask_dir=Config.TRAIN_MASK_DIR,
        test_image_dir=Config.TEST_IMAGE_DIR,
        test_mask_dir=Config.TEST_MASK_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        target_size=(Config.INPUT_HEIGHT, Config.INPUT_WIDTH)
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED
    ).to(device)
    
    # Load trained model weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print original model statistics
    total_params, trainable_params = count_parameters(model)
    original_size = get_model_size(model)
    
    print(f"\nOriginal Model Statistics:")
    print(f"Parameters: {total_params:,}")
    print(f"Model size: {original_size:.2f} MB")
    
    # Evaluate original model
    criterion = CombinedLoss(
        dice_weight=Config.DICE_WEIGHT,
        ce_weight=Config.BCE_WEIGHT
    )
    
    print("\nEvaluating original model...")
    original_metrics = evaluate_model(model, val_loader, criterion, device)
    print_metrics(original_metrics, "Original ")
    
    # Create pruner
    pruner = ModelPruner(
        model=model,
        pruning_amount=args.pruning_amount,
        structured=args.structured
    )
    
    # Apply pruning
    pruner.prune_model()
    
    # Get sparsity information
    sparsity_info = pruner.get_sparsity_info()
    print(f"\nSparsity Information:")
    print(f"Total parameters: {sparsity_info['total_params']:,}")
    print(f"Zero parameters: {sparsity_info['zero_params']:,}")
    print(f"Sparsity: {sparsity_info['sparsity']:.1%}")
    print(f"Compression ratio: {sparsity_info['compression_ratio']:.2f}x")
    
    # Evaluate pruned model (before fine-tuning)
    print("\nEvaluating pruned model (before fine-tuning)...")
    pruned_metrics = evaluate_model(model, val_loader, criterion, device)
    print_metrics(pruned_metrics, "Pruned ")
    
    # Fine-tune pruned model if requested
    if args.fine_tune:
        model = fine_tune_pruned_model(
            model, train_loader, val_loader, device, 
            epochs=args.fine_tune_epochs
        )
        
        # Evaluate fine-tuned model
        print("\nEvaluating fine-tuned pruned model...")
        final_metrics = evaluate_model(model, val_loader, criterion, device)
        print_metrics(final_metrics, "Fine-tuned ")
    else:
        final_metrics = pruned_metrics
    
    # Remove pruning masks for final model
    if args.remove_masks:
        pruner.remove_pruning_masks()
        final_size = get_model_size(model)
        print(f"\nFinal model size after removing masks: {final_size:.2f} MB")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PRUNING SUMMARY")
    print("=" * 50)
    print(f"Pruning method: {'Structured' if args.structured else 'Unstructured'}")
    print(f"Pruning amount: {args.pruning_amount:.1%}")
    print(f"Original IoU: {original_metrics.get('mean_iou', 0):.4f}")
    print(f"Pruned IoU: {pruned_metrics.get('mean_iou', 0):.4f}")
    print(f"Final IoU: {final_metrics.get('mean_iou', 0):.4f}")
    
    iou_drop = original_metrics.get('mean_iou', 0) - final_metrics.get('mean_iou', 0)
    print(f"IoU drop: {iou_drop:.4f} ({iou_drop/original_metrics.get('mean_iou', 1)*100:.1f}%)")
    print(f"Compression ratio: {sparsity_info['compression_ratio']:.2f}x")
    
    # Save pruned model
    save_path = args.output_path
    if save_path:
        save_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=0,
            best_metric=final_metrics.get('mean_iou', 0),
            checkpoint_dir=os.path.dirname(save_path),
            filename=os.path.basename(save_path)
        )
        print(f"\nPruned model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prune semantic segmentation model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output-path', type=str,
                        default='train/checkpoints/pruned_model.pth',
                        help='Path to save pruned model')
    parser.add_argument('--pruning-amount', type=float, default=0.3,
                        help='Fraction of parameters to prune (default: 0.3)')
    parser.add_argument('--structured', action='store_true',
                        help='Use structured pruning instead of unstructured')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune model after pruning')
    parser.add_argument('--fine-tune-epochs', type=int, default=20,
                        help='Number of fine-tuning epochs (default: 20)')
    parser.add_argument('--remove-masks', action='store_true',
                        help='Remove pruning masks from final model')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0 < args.pruning_amount < 1):
        raise ValueError("Pruning amount must be between 0 and 1")
    
    try:
        main(args)
    except Exception as e:
        print(f"Pruning failed with error: {e}")
        raise
