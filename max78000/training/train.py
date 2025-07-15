"""
Training Script for EMG Classification Model
Trains models for deployment on MAX78000 hardware
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model_architecture import create_model, CLASS_NAMES
from dataset import EMGDataProcessor, load_emg_data, print_dataset_info


class EMGTrainer:
    """
    Trainer class for EMG classification models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 log_dir: str = "logs",
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Training device
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                batch_acc = 100. * correct / total
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), 
                                     self.epoch * len(train_loader) + batch_idx)
                self.writer.add_scalar('Train/Batch_Accuracy', batch_acc,
                                     self.epoch * len(train_loader) + batch_idx)
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, 
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy, predictions, targets)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 10) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min',
                                                       patience=patience//2,
                                                       factor=0.5,
                                                       verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_acc)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Final evaluation
        final_report = self.evaluate_model(val_loader, val_preds, val_targets)
        
        # Training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'final_report': final_report
        }
        
        return history
    
    def evaluate_model(self, 
                      data_loader: DataLoader,
                      predictions: np.ndarray,
                      targets: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            data_loader: Data loader
            predictions: Predicted labels
            targets: True labels
            
        Returns:
            Evaluation metrics
        """
        # Classification report
        class_report = classification_report(targets, predictions, 
                                           target_names=CLASS_NAMES,
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Overall metrics
        overall_acc = np.sum(cm.diagonal()) / np.sum(cm)
        
        print("\nEvaluation Results:")
        print("-" * 50)
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print("\nPer-class Accuracy:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_name}: {per_class_acc[i]:.4f}")
        
        return {
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': per_class_acc.tolist(),
            'overall_accuracy': overall_acc
        }
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Save best model
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_filepath)
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Epoch: {self.epoch}, Best Val Acc: {self.best_val_acc:.2f}%")
    
    def plot_training_history(self):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_history.png'))
        plt.close()


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train EMG Classification Model')
    parser.add_argument('--model', type=str, default='standard',
                       choices=['standard', 'lightweight', 'quantized'],
                       help='Model architecture')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Select device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(args.model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data
    processor = EMGDataProcessor(
        normalize_features=True,
        augment_data=True,
        noise_std=0.01
    )
    
    if args.data_path and os.path.exists(args.data_path):
        features, labels = load_emg_data(args.data_path)
        print(f"Loaded data from {args.data_path}")
    else:
        print("Generating synthetic data for training...")
        features, labels = processor.generate_synthetic_data(
            n_samples=5000,
            n_features=72,
            n_classes=8
        )
    
    # Prepare datasets
    datasets = processor.prepare_dataset(features, labels)
    print_dataset_info(datasets)
    
    # Create dataloaders
    dataloaders = processor.create_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Setup trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/emg_training_{args.model}_{timestamp}"
    checkpoint_dir = f"checkpoints/emg_training_{args.model}_{timestamp}"
    
    trainer = EMGTrainer(model, device, log_dir, checkpoint_dir)
    
    # Train model
    history = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save training history
    history_path = os.path.join(log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save preprocessor
    processor_path = os.path.join(checkpoint_dir, 'preprocessor.pkl')
    processor.save_preprocessor(processor_path)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {checkpoint_dir}/best_model.pth")
    print(f"Training logs saved to: {log_dir}")
    print(f"Preprocessor saved to: {processor_path}")


if __name__ == "__main__":
    main()