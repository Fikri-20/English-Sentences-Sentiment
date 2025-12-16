import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUClassifier(nn.Module):
    """GRU-based text classifier"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 bidirectional: bool = True,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize GRU classifier
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of GRU hidden state
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            pretrained_embeddings: Pretrained word embeddings
        """
        super(GRUClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # GRU layer
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(gru_output_dim, num_classes)
    
    def forward(self, text, lengths):
        """
        Forward pass
        
        Args:
            text: Input text tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # GRU
        packed_output, hidden = self.gru(packed)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layer
        output = self.fc(hidden)
        
        return output


class BiGRUAttention(nn.Module):
    """Bidirectional GRU with attention mechanism"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize BiGRU with Attention
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of GRU hidden state
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout probability
            pretrained_embeddings: Pretrained word embeddings
        """
        super(BiGRUAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def attention_layer(self, gru_output, lengths):
        """
        Apply attention mechanism
        
        Args:
            gru_output: GRU output [batch_size, seq_len, hidden_dim*2]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Context vector [batch_size, hidden_dim*2] and attention weights
        """
        # Calculate attention scores
        attention_scores = self.attention(gru_output).squeeze(-1)
        
        # Create mask for padding
        batch_size, max_len = gru_output.size(0), gru_output.size(1)
        mask = torch.arange(max_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)
        
        # Apply mask
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), gru_output).squeeze(1)
        
        return context, attention_weights
    
    def forward(self, text, lengths):
        """
        Forward pass
        
        Args:
            text: Input text tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # GRU
        packed_output, hidden = self.gru(packed)
        
        # Unpack
        gru_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention
        context, attention_weights = self.attention_layer(gru_output, lengths)
        
        # Dropout
        context = self.dropout(context)
        
        # Fully connected layer
        output = self.fc(context)
        
        return output


class StackedGRU(nn.Module):
    """Stacked GRU with residual connections"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize Stacked GRU with residual connections
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of GRU hidden state
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout probability
            pretrained_embeddings: Pretrained word embeddings
        """
        super(StackedGRU, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # First GRU layer
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        
        # Additional GRU layers with residual connections
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)
            for _ in range(num_layers - 1)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, text, lengths):
        """
        Forward pass
        
        Args:
            text: Input text tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # First GRU layer
        packed_output, hidden = self.gru1(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.layer_norms[0](output)
        
        # Additional layers with residual connections
        for i, gru_layer in enumerate(self.gru_layers):
            residual = output
            
            # Pack again
            packed = pack_padded_sequence(output, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = gru_layer(packed)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            
            # Add residual connection
            output = output + residual
            output = self.layer_norms[i + 1](output)
            output = self.dropout(output)
        
        # Use last hidden state (take from last time step for each sequence)
        batch_size = output.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_outputs.append(output[i, lengths[i]-1, :])
        
        last_output = torch.stack(last_outputs)
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)
        
        return output


if __name__ == "__main__":
    # Test models
    batch_size = 8
    seq_len = 50
    vocab_size = 10000
    
    # Generate random data
    text = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(10, seq_len, (batch_size,))
    
    print("Testing GRU Classifier...")
    gru_model = GRUClassifier(vocab_size=vocab_size, num_classes=3)
    gru_output = gru_model(text, lengths)
    print(f"GRU Output shape: {gru_output.shape}")
    
    print("\nTesting BiGRU with Attention...")
    bigru_model = BiGRUAttention(vocab_size=vocab_size, num_classes=3)
    bigru_output = bigru_model(text, lengths)
    print(f"BiGRU+Attention Output shape: {bigru_output.shape}")
    
    print("\nTesting Stacked GRU...")
    stacked_model = StackedGRU(vocab_size=vocab_size, num_classes=3)
    stacked_output = stacked_model(text, lengths)
    print(f"Stacked GRU Output shape: {stacked_output.shape}")
    
    print("\n✅ All GRU models initialized successfully!")
