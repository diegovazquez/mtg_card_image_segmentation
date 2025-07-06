import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Import local modules
from model import LiteHRNet, create_model, save_model
from dataset import create_dataloaders
from metrics import CornerMetrics, CornerLoss, EarlyStopping


class Trainer:
    """
    Main trainer class for corner detection model
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict[str, Any],
                 resume_from: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
            config: Training configuration
            resume_from: Path to checkpoint to resume from (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['lr_factor'],
            patience=config['lr_patience']
        )
        
        # Initialize loss function
        self.criterion = CornerLoss(
            image_size=config['image_size'],
            heatmap_size=config['heatmap_size']
        )
        
        # Initialize metrics
        self.train_metrics = CornerMetrics(image_size=config['image_size'])
        self.val_metrics = CornerMetrics(image_size=config['image_size'])
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            mode='min',
            restore_best_weights=True
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler('cuda')
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Create output directory
        self.output_dir = os.path.realpath(config['output_dir'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(device_type="cuda"):
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            self.train_metrics.update(predictions, targets)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        
        return avg_loss, metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch
        
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                with autocast(device_type="cuda"):
                    predictions = self.model(images)
                    loss = self.criterion(predictions, targets)
                
                # Update metrics
                self.val_metrics.update(predictions, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        
        return avg_loss, metrics
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint and resume training state
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint_path = os.path.realpath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict'")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        
        if 'loss' in checkpoint:
            self.best_val_loss = checkpoint['loss']
        
        # Load training history if available
        if 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
            if 'history' in checkpoint['metrics']:
                self.history = checkpoint['metrics']['history']
            if 'best_val_loss' in checkpoint['metrics']:
                self.best_val_loss = checkpoint['metrics']['best_val_loss']
        
        print(f"Resumed from epoch {self.epoch + 1}, best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, 
                       filename: str,
                       is_best: bool = False,
                       additional_info: Optional[Dict] = None):
        """
        Save model checkpoint
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
            additional_info: Additional information to save
        """
        checkpoint_path = os.path.join(self.output_dir, filename)
        
        checkpoint_info = {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        if additional_info:
            checkpoint_info.update(additional_info)
        
        save_model(
            model=self.model,
            checkpoint_path=checkpoint_path,
            optimizer=self.optimizer,
            epoch=self.epoch,
            loss=self.best_val_loss,
            metrics=checkpoint_info
        )
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            save_model(
                model=self.model,
                checkpoint_path=best_path,
                optimizer=self.optimizer,
                epoch=self.epoch,
                loss=self.best_val_loss,
                metrics=checkpoint_info
            )
    
    def train(self):
        """
        Main training loop
        """
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Validate epoch
            val_loss, val_metrics = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Train Metrics: {train_metrics}")
            print(f"Val Metrics: {val_metrics}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(
                filename=f'checkpoint_epoch_{epoch + 1}.pth',
                is_best=is_best,
                additional_info={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
            )
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(
            filename='final_model.pth',
            additional_info={'training_time': training_time}
        )
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_history = convert_numpy_types(self.history)
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        return self.history


def get_default_config() -> Dict[str, Any]:
    """
    Get default training configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'num_epochs': 200,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'early_stopping_patience': 20,
        'image_size': (480, 640),
        'heatmap_size': (160, 120),
        'num_workers': 4,
        'output_dir': './outputs',
        'dataset_path': '../dataset'
    }


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train corner detection model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_default_config()
    
    if args.config:
        config_path = os.path.realpath(args.config)
        with open(config_path, 'r') as f:
            config.update(json.load(f))
    
    # Override with command line arguments
    if args.dataset:
        config['dataset_path'] = os.path.realpath(args.dataset)
    if args.output:
        config['output_dir'] = os.path.realpath(args.output)
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        dataset_path=config['dataset_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size'],
        heatmap_size=config['heatmap_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(num_keypoints=4, pretrained=True, heatmap_size=config['heatmap_size'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        resume_from=args.resume
    )
    
    # Start training
    try:
        history = trainer.train()
        
        # Export model for inference
        print("Exporting model for inference...")
        export_model(model, config['output_dir'], device)
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint('interrupted_model.pth')
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


def export_model(model: nn.Module, output_dir: str, device: torch.device):
    """
    Export model for inference and deployment
    
    Args:
        model: Trained model
        output_dir: Output directory
        device: Device model is on
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 480).to(device)
    
    # Export to TorchScript
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        script_path = os.path.join(output_dir, 'model_traced.pt')
        traced_model.save(script_path)
        print(f"TorchScript model saved to: {script_path}")
    except Exception as e:
        print(f"Failed to export TorchScript model: {e}")
    


if __name__ == '__main__':
    main()