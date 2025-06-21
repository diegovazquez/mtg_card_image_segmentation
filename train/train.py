"""
Main training script for semantic segmentation.
Features FP16 mixed precision training, comprehensive logging, and model checkpointing.
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import wandb

from config import Config
from model import create_model, count_parameters, get_model_size
from dataset import create_dataloaders
from utils import (
    CombinedLoss, MetricsCalculator, save_checkpoint, load_checkpoint,
    plot_training_history, print_metrics
)

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change in metric to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = 0
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, metric, model):
        """
        Check if training should stop.
        
        Args:
            metric (float): Current validation metric
            model (nn.Module): Model to potentially save weights from
            
        Returns:
            bool: Whether to stop training
        """
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        epoch (int): Current epoch number
        
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    metrics_calc = MetricsCalculator(num_classes=Config.NUM_CLASSES, device=device)
    
    total_batches = len(train_loader)
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast('cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Mixed precision backward pass
        if Config.USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            metrics_calc.update(loss, outputs, masks)
        
        # Print progress
        if batch_idx % 20 == 0:
            batch_time = time.time() - start_time
            eta = batch_time * (total_batches - batch_idx) / (batch_idx + 1)
            print(f'Epoch {epoch} [{batch_idx:4d}/{total_batches}] '
                  f'Loss: {loss.item():.4f} '
                  f'ETA: {eta/60:.1f}min')
    
    return metrics_calc.get_metrics()

def validate_epoch(model, val_loader, criterion, device, epoch):
    """
    Validate model for one epoch.
    
    Args:
        model (nn.Module): Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch (int): Current epoch number
        
    Returns:
        dict: Validation metrics for the epoch
    """
    model.eval()
    metrics_calc = MetricsCalculator(num_classes=Config.NUM_CLASSES, device=device)
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast('cuda', enabled=Config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Update metrics
            metrics_calc.update(loss, outputs, masks)
    
    return metrics_calc.get_metrics()

def create_optimizer(model, config):
    """
    Create optimizer based on configuration.
    
    Args:
        model (nn.Module): Model to optimize
        config: Configuration object
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if config.OPTIMIZER.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")

def create_scheduler(optimizer, config, train_loader):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration object
        train_loader: Training data loader
        
    Returns:
        Learning rate scheduler
    """
    if config.SCHEDULER.lower() == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.LEARNING_RATE * 0.01
        )
    elif config.SCHEDULER.lower() == 'cosine_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.NUM_EPOCHS // 4,
            T_mult=2
        )
    else:
        return None

def main(args):
    """Main training function."""
    
    # Print configuration
    Config.print_config()
    
    # Create directories
    Config.create_directories()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="mtg-card-segmentation",
            config=vars(Config),
            name=f"lraspp_{int(time.time())}"
        )
    
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
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED
    ).to(device)
    
    # Print model statistics
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model size: {model_size:.2f} MB")
    
    # Create loss function
    criterion = CombinedLoss(
        dice_weight=Config.DICE_WEIGHT,
        ce_weight=Config.BCE_WEIGHT
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, Config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, Config, train_loader)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler('cuda', enabled=Config.USE_AMP)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE)
    
    # Training history
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from {args.resume}")
        start_epoch, best_metric = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
        start_epoch += 1
    
    print("Starting training...")
    total_start_time = time.time()
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validation
        if epoch % Config.VALIDATE_EVERY == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch
            )
        else:
            val_metrics = {'loss': 0, 'mean_iou': 0}
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS-1} - {epoch_time:.1f}s")
        print_metrics(train_metrics, "Train ")
        if epoch % Config.VALIDATE_EVERY == 0:
            print_metrics(val_metrics, "Val ")
        
        # Log to wandb
        if args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_iou': train_metrics['mean_iou'],
                'train_dice': train_metrics['mean_dice'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            if epoch % Config.VALIDATE_EVERY == 0:
                log_dict.update({
                    'val_loss': val_metrics['loss'],
                    'val_iou': val_metrics['mean_iou'],
                    'val_dice': val_metrics['mean_dice']
                })
            wandb.log(log_dict)
        
        # Save checkpoint
        current_metric = val_metrics.get('mean_iou', 0)
        is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric,
                Config.CHECKPOINT_DIR, 'best_model.pth'
            )
        
        if epoch % Config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_metric,
                Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'
            )
        
        # Early stopping check
        if early_stopping(current_metric, model):
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        print("-" * 50)
    
    # Training completed
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best validation IoU: {best_metric:.4f}")
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, epoch, best_metric,
        Config.CHECKPOINT_DIR, 'final_model.pth'
    )
    
    # Plot training history
    try:
        plot_path = os.path.join(Config.LOG_DIR, 'training_history.png')
        plot_training_history(
            train_losses, val_losses, train_metrics_history, val_metrics_history,
            save_path=plot_path
        )
    except Exception as e:
        print(f"Could not save training plot: {e}")
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train semantic segmentation model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file (optional)')
    
    args = parser.parse_args()
    
    # Override config if custom config provided
    if args.config and os.path.exists(args.config):
        # Load custom config (implement if needed)
        pass
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
