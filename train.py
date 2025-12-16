import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
from tqdm import tqdm
import json

# Import custom modules
from utils.preprocessing import TextPreprocessor
from utils.dataset import create_dataloaders, load_data_from_csv, Vocabulary
from utils.metrics import MetricsCalculator, plot_training_history
from models.lstm import LSTMClassifier, BiLSTMAttention, MultiTaskLSTM
from models.gru import GRUClassifier, BiGRUAttention


class Trainer:
    """Training class for sentiment analysis models"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 device,
                 num_classes=3,
                 checkpoint_dir='checkpoints',
                 log_dir='runs'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            num_classes: Number of classes
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(num_classes=num_classes)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            text = batch['text'].to(self.device)
            labels = batch['label'].to(self.device).squeeze()
            lengths = batch['length']
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(text, lengths)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for batch in pbar:
                # Move to device
                text = batch['text'].to(self.device)
                labels = batch['label'].to(self.device).squeeze()
                lengths = batch['length']
                
                # Forward pass
                outputs = self.model(text, lengths)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, num_epochs, scheduler=None):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Learning rate scheduler
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print("-" * 80)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc, train_preds, train_labels = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # Calculate detailed metrics
            val_metrics = self.metrics_calculator.calculate_metrics(
                np.array(val_labels), 
                np.array(val_preds)
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print("-" * 80)
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        fig = plot_training_history(self.history)
        fig.savefig(os.path.join(self.checkpoint_dir, 'training_history.png'))
        
        self.writer.close()
        
        print(f"\n🎉 Training completed!")
        print(f"Best Val Acc: {self.best_val_acc:.4f} | Best Val Loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Sentiment Analysis Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--text_column', type=str, default='text', help='Name of text column')
    parser.add_argument('--label_column', type=str, default='label', help='Name of label column')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='bilstm_attention',
                       choices=['lstm', 'bilstm_attention', 'gru', 'bigru_attention'],
                       help='Model architecture')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    
    # Other arguments
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='runs', help='Tensorboard log directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("Sentiment Analysis Training")
    print("=" * 80)
    
    # Load and preprocess data
    print("\n📂 Loading data...")
    texts, labels = load_data_from_csv(args.data_path, args.text_column, args.label_column)
    print(f"Loaded {len(texts)} samples")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    # Create dataloaders
    print("\n🔄 Creating dataloaders...")
    train_loader, val_loader, vocab = create_dataloaders(
        train_texts, train_labels,
        val_texts, val_labels,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Save vocabulary
    vocab.save(os.path.join(args.checkpoint_dir, 'vocabulary.pkl'))
    print(f"Vocabulary size: {len(vocab)}")
    
    # Initialize model
    print(f"\n🏗️ Initializing {args.model_type} model...")
    
    if args.model_type == 'lstm':
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            bidirectional=False
        )
    elif args.model_type == 'bilstm_attention':
        model = BiLSTMAttention(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout
        )
    elif args.model_type == 'gru':
        model = GRUClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            bidirectional=True
        )
    elif args.model_type == 'bigru_attention':
        model = BiGRUAttention(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        num_classes=args.num_classes,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs, scheduler=scheduler)


if __name__ == "__main__":
    main()
