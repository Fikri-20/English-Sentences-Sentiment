#!/usr/bin/env python3
"""
Interactive Sentiment Analysis - Type sentences and get instant predictions
"""
import torch
import pickle
import re
import string

# Load vocabulary and model
print("Loading model...")
with open('checkpoints/vocabulary.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

word2idx = vocab_dict['word2idx']
vocab_size = len(word2idx)

checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')

from models.lstm import BiLSTMAttention
model = BiLSTMAttention(
    vocab_size=vocab_size,
    embedding_dim=300,
    hidden_dim=256,
    num_layers=2,
    num_classes=3,
    dropout=0.5
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Sentiment labels
labels = {0: "😡 Negative", 1: "😐 Neutral", 2: "😊 Positive"}

def preprocess(text):
    """Simple text preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text.strip()

def encode_text(text, word2idx, max_len=50):
    """Convert text to indices"""
    tokens = text.split()
    encoded = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    return encoded[:max_len]

def predict(text):
    """Predict sentiment for a given text"""
    # Preprocess
    cleaned = preprocess(text)
    encoded = encode_text(cleaned, word2idx)
    
    if not encoded:
        return None, None
    
    # Convert to tensor
    text_tensor = torch.tensor([encoded])
    lengths = torch.tensor([len(encoded)])
    
    # Predict
    with torch.no_grad():
        output = model(text_tensor, lengths)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
    
    return pred, probs

# Main interactive loop
print("\n" + "="*70)
print("🎭 SENTIMENT ANALYSIS - Interactive Mode")
print("="*70)
print("Type a sentence and press Enter to analyze sentiment")
print("Commands: 'quit', 'exit', or 'q' to stop")
print("="*70 + "\n")

while True:
    # Get user input
    try:
        text = input("📝 Your text: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Goodbye!")
        break
    
    if text.lower() in ['quit', 'exit', 'q', '']:
        print("\n👋 Goodbye!")
        break
    
    # Make prediction
    pred, probs = predict(text)
    
    if pred is None:
        print("⚠️  Could not analyze (empty text after cleaning)\n")
        continue
    
    # Display results
    print(f"\n🎯 Sentiment: {labels[pred]}")
    print(f"📊 Confidence Scores:")
    print(f"   Negative: {'█' * int(probs[0]*20)} {probs[0]*100:5.1f}%")
    print(f"   Neutral:  {'█' * int(probs[1]*20)} {probs[1]*100:5.1f}%")
    print(f"   Positive: {'█' * int(probs[2]*20)} {probs[2]*100:5.1f}%")
    print("-" * 70 + "\n")
