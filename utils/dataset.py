import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import pickle

class Vocabulary:
    """Vocabulary class for text tokenization and numericalization"""
    
    def __init__(self, min_freq: int = 2, max_size: int = None):
        """
        Initialize vocabulary
        
        Args:
            min_freq: Minimum frequency for a word to be included
            max_size: Maximum vocabulary size
        """
        self.min_freq = min_freq
        self.max_size = max_size
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        # Token to index mappings
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.word_freq = Counter()
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from texts
        
        Args:
            texts: List of texts
        """
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # Filter by frequency
        valid_words = [
            word for word, freq in self.word_freq.items() 
            if freq >= self.min_freq
        ]
        
        # Sort by frequency (most common first)
        valid_words = sorted(valid_words, key=lambda w: self.word_freq[w], reverse=True)
        
        # Limit vocabulary size
        if self.max_size is not None:
            valid_words = valid_words[:self.max_size - len(self.word2idx)]
        
        # Add words to vocabulary
        for word in valid_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of indices
        
        Args:
            text: Input text
            
        Returns:
            List of token indices
        """
        words = text.split()
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        """
        Convert list of indices to text
        
        Args:
            indices: List of token indices
            
        Returns:
            Decoded text
        """
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'min_freq': self.min_freq,
                'max_size': self.max_size
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(min_freq=data['min_freq'], max_size=data['max_size'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = data['word_freq']
        
        return vocab


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis"""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 vocab: Vocabulary = None,
                 max_length: int = 100):
        """
        Initialize dataset
        
        Args:
            texts: List of texts
            labels: List of labels
            vocab: Vocabulary object (if None, will be built)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Build or use existing vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocabulary(texts)
        else:
            self.vocab = vocab
        
        # Encode texts
        self.encoded_texts = [self.vocab.encode(text) for text in texts]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Returns:
            Dictionary with 'text', 'label', and 'length'
        """
        encoded = self.encoded_texts[idx]
        
        # Truncate if necessary
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        
        return {
            'text': torch.LongTensor(encoded),
            'label': torch.LongTensor([self.labels[idx]]),
            'length': len(encoded)
        }


class EmotionDataset(Dataset):
    """PyTorch Dataset for multi-label emotion detection"""
    
    def __init__(self, 
                 texts: List[str], 
                 emotions: List[List[int]],
                 vocab: Vocabulary = None,
                 max_length: int = 100):
        """
        Initialize dataset
        
        Args:
            texts: List of texts
            emotions: List of emotion labels (multi-label)
            vocab: Vocabulary object
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.emotions = emotions
        self.max_length = max_length
        
        # Build or use existing vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocabulary(texts)
        else:
            self.vocab = vocab
        
        # Encode texts
        self.encoded_texts = [self.vocab.encode(text) for text in texts]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        
        # Truncate if necessary
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        
        return {
            'text': torch.LongTensor(encoded),
            'emotions': torch.FloatTensor(self.emotions[idx]),
            'length': len(encoded)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable length sequences
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with padded tensors
    """
    texts = [item['text'] for item in batch]
    labels = torch.cat([item['label'] for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return {
        'text': texts_padded,
        'label': labels,
        'length': lengths
    }


def collate_fn_emotions(batch):
    """
    Collate function for emotion detection (multi-label)
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with padded tensors
    """
    texts = [item['text'] for item in batch]
    emotions = torch.stack([item['emotions'] for item in batch])
    lengths = torch.LongTensor([item['length'] for item in batch])
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return {
        'text': texts_padded,
        'emotions': emotions,
        'length': lengths
    }


def create_dataloaders(train_texts: List[str],
                       train_labels: List[int],
                       val_texts: List[str],
                       val_labels: List[int],
                       batch_size: int = 32,
                       max_length: int = 100,
                       num_workers: int = 2) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Create train and validation dataloaders
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, vocabulary)
    """
    # Build vocabulary from training data
    vocab = Vocabulary()
    vocab.build_vocabulary(train_texts)
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, vocab


def load_data_from_csv(csv_path: str, 
                       text_column: str = 'text',
                       label_column: str = 'label') -> Tuple[List[str], List[int]]:
    """
    Load data from CSV file
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(csv_path)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    return texts, labels


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This movie is really great",
        "I am happy today",
        "This is very bad",
        "The weather is beautiful",
    ]
    sample_labels = [1, 1, 0, 1]  # 1: positive, 0: negative
    
    # Create vocabulary
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary(sample_texts)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    dataset = SentimentDataset(sample_texts, sample_labels, vocab)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Test batch
    for batch in dataloader:
        print(f"Text shape: {batch['text'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        print(f"Lengths: {batch['length']}")
        break
