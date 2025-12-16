#!/usr/bin/env python3
"""
Download HuggingFace Dataset: Sp1786/multiclass-sentiment-analysis-dataset
"""
from datasets import load_dataset
import pandas as pd
import os

print("📥 Downloading dataset from HuggingFace...")
print("Dataset: Sp1786/multiclass-sentiment-analysis-dataset")
print("This may take a few minutes...\n")

# Download dataset
dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

print("✅ Dataset downloaded successfully!")
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}\n")

# Convert to pandas DataFrames
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Save to CSV
os.makedirs('data/raw', exist_ok=True)
train_path = 'data/raw/huggingface_sentiment_train.csv'
test_path = 'data/raw/huggingface_sentiment_test.csv'

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ Saved training data to: {train_path}")
print(f"✅ Saved test data to: {test_path}\n")

# Display dataset info
print("="*70)
print("📊 Dataset Information")
print("="*70)
print(f"\nTraining Set: {len(train_df)} samples")
print(train_df['label'].value_counts().sort_index())
print(f"\nTest Set: {len(test_df)} samples")
print(test_df['label'].value_counts().sort_index())
print("\n" + "="*70)
print("Sample data:")
print(train_df.head(3))
print("="*70)
