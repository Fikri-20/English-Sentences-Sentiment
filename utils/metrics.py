import numpy as np
import torch
from sklearn.metrics import ( # pyright: ignore[reportMissingModuleSource]
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    """Calculate and track metrics for model evaluation"""
    
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes (for visualization)
        """
        self.num_classes = num_classes
        
        if class_names is None:
            if num_classes == 2:
                self.class_names = ['Negative', 'Positive']
            elif num_classes == 3:
                self.class_names = ['Negative', 'Neutral', 'Positive']
            else:
                self.class_names = [f'Class_{i}' for i in range(num_classes)]
        else:
            self.class_names = class_names
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy for multi-class
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # Per-class metrics
        if self.num_classes > 2:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, class_name in enumerate(self.class_names):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             save_path: str = None,
                             normalize: bool = True) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
            normalize: Whether to normalize confusion matrix
            
        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_classification_report(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> str:
        """
        Get detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred, 
                                    target_names=self.class_names,
                                    zero_division=0)
    
    def plot_metrics_comparison(self, 
                               metrics_dict: Dict[str, Dict[str, float]],
                               save_path: str = None) -> plt.Figure:
        """
        Plot comparison of metrics across different models
        
        Args:
            metrics_dict: Dictionary of {model_name: {metric_name: value}}
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        models = list(metrics_dict.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_names):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            axes[idx].bar(models, values, color='skyblue', edgecolor='navy')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} Comparison')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', 
                             ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class MultiLabelMetrics:
    """Metrics for multi-label classification (emotion detection)"""
    
    def __init__(self, emotion_names: List[str]):
        """
        Initialize multi-label metrics calculator
        
        Args:
            emotion_names: Names of emotions
        """
        self.emotion_names = emotion_names
    
    def hamming_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Hamming loss
        
        Args:
            y_true: True labels [batch_size, num_emotions]
            y_pred: Predicted labels [batch_size, num_emotions]
            
        Returns:
            Hamming loss
        """
        return np.mean(y_true != y_pred)
    
    def subset_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate subset accuracy (exact match)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Subset accuracy
        """
        return np.mean(np.all(y_true == y_pred, axis=1))
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate multi-label metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Threshold for binary predictions
            
        Returns:
            Dictionary with metrics
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > threshold).astype(int)
        
        metrics = {
            'hamming_loss': self.hamming_loss(y_true, y_pred_binary),
            'subset_accuracy': self.subset_accuracy(y_true, y_pred_binary),
        }
        
        # Per-emotion metrics
        for i, emotion in enumerate(self.emotion_names):
            metrics[f'accuracy_{emotion}'] = accuracy_score(y_true[:, i], y_pred_binary[:, i])
            metrics[f'precision_{emotion}'] = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            metrics[f'recall_{emotion}'] = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            metrics[f'f1_{emotion}'] = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        
        return metrics
    
    def plot_per_emotion_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                threshold: float = 0.5,
                                save_path: str = None) -> plt.Figure:
        """
        Plot per-emotion metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Threshold for binary predictions
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        y_pred_binary = (y_pred > threshold).astype(int)
        
        emotions = self.emotion_names
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        
        # Calculate metrics for each emotion
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i in range(len(emotions)):
            precision_scores.append(precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0))
            recall_scores.append(recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0))
            f1_scores.append(f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0))
        
        # Create plot
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision_scores, width, label='Precision', color='skyblue')
        ax.bar(x, recall_scores, width, label='Recall', color='lightgreen')
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightcoral')
        
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Score')
        ax.set_title('Per-Emotion Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: str = None) -> plt.Figure:
    """
    Plot training history (loss and metrics over epochs)
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 1, 2, 1, 0, 2, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 2])
    
    calculator = MetricsCalculator(num_classes=3)
    
    # Calculate metrics
    metrics = calculator.calculate_metrics(y_true, y_pred)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(calculator.get_classification_report(y_true, y_pred))
