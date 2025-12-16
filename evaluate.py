import torch
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm

from utils.dataset import SentimentDataset, Vocabulary, collate_fn
from utils.metrics import MetricsCalculator
from torch.utils.data import DataLoader

# Import models
from models.lstm import LSTMClassifier, BiLSTMAttention
from models.gru import GRUClassifier, BiGRUAttention


def load_model(model_path, model_type, vocab_size, num_classes=3, device='cpu'):
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        vocab_size: Size of vocabulary
        num_classes: Number of classes
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Initialize model
    if model_type == 'lstm':
        model = LSTMClassifier(vocab_size=vocab_size, num_classes=num_classes, bidirectional=False)
    elif model_type == 'bilstm_attention':
        model = BiLSTMAttention(vocab_size=vocab_size, num_classes=num_classes)
    elif model_type == 'gru':
        model = GRUClassifier(vocab_size=vocab_size, num_classes=num_classes)
    elif model_type == 'bigru_attention':
        model = BiGRUAttention(vocab_size=vocab_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, data_loader, device, num_classes=3):
    """
    Evaluate model on data
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device
        num_classes: Number of classes
        
    Returns:
        Dictionary with predictions and labels
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            text = batch['text'].to(device)
            labels = batch['label'].to(device).squeeze()
            lengths = batch['length']
            
            # Forward pass
            outputs = model(text, lengths)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Sentiment Analysis Model')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data CSV')
    
    # Optional arguments
    parser.add_argument('--model_type', type=str, default='bilstm_attention',
                       choices=['lstm', 'bilstm_attention', 'gru', 'bigru_attention'])
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--label_column', type=str, default='label')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Sentiment Analysis - Model Evaluation")
    print("=" * 80)
    
    # Load vocabulary
    print(f"\n📂 Loading vocabulary from {args.vocab_path}...")
    vocab = Vocabulary.load(args.vocab_path)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load test data
    print(f"\n📂 Loading test data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    texts = df[args.text_column].tolist()
    labels = df[args.label_column].tolist()
    print(f"Test samples: {len(texts)}")
    
    # Create dataset and dataloader
    test_dataset = SentimentDataset(texts, labels, vocab, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    # Load model
    print(f"\n🏗️ Loading {args.model_type} model from {args.model_path}...")
    model, checkpoint = load_model(
        args.model_path, 
        args.model_type, 
        len(vocab), 
        args.num_classes,
        args.device
    )
    
    print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")
    
    # Evaluate
    print(f"\n🔍 Evaluating model on test set...")
    results = evaluate_model(model, test_loader, args.device, args.num_classes)
    
    # Calculate metrics
    print(f"\n📊 Calculating metrics...")
    metrics_calc = MetricsCalculator(num_classes=args.num_classes)
    metrics = metrics_calc.calculate_metrics(results['labels'], results['predictions'])
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    # Print classification report
    print("\n" + "-" * 80)
    print("CLASSIFICATION REPORT")
    print("-" * 80)
    print(metrics_calc.get_classification_report(results['labels'], results['predictions']))
    
    # Plot confusion matrix
    print("\n📈 Generating confusion matrix...")
    cm_fig = metrics_calc.plot_confusion_matrix(
        results['labels'], 
        results['predictions'],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png'),
        normalize=True
    )
    print(f"Saved confusion matrix to {args.output_dir}/confusion_matrix.png")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'text': texts,
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'correct': results['labels'] == results['predictions']
    })
    
    # Add probability columns
    for i in range(args.num_classes):
        results_df[f'prob_class_{i}'] = results['probabilities'][:, i]
    
    results_path = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Saved predictions to {results_path}")
    
    # Save metrics to JSON
    import json
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Saved metrics to {metrics_path}")
    
    # Error analysis
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    incorrect = results_df[~results_df['correct']]
    print(f"\nTotal errors: {len(incorrect)} / {len(results_df)} ({len(incorrect)/len(results_df)*100:.2f}%)")
    
    if len(incorrect) > 0:
        print("\nSample errors:")
        print("-" * 80)
        for idx, row in incorrect.head(5).iterrows():
            print(f"\nText: {row['text'][:100]}...")
            print(f"True: {row['true_label']} | Predicted: {row['predicted_label']}")
            probs = [f"{row[f'prob_class_{i}']:.3f}" for i in range(args.num_classes)]
            print(f"Probabilities: {probs}")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
