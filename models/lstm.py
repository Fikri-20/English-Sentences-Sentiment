import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    """LSTM-based text classifier for sentiment analysis"""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 bidirectional: bool = False,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize LSTM classifier
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            pretrained_embeddings: Pretrained word embeddings
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # LSTM layer
        self.lstm = nn.LSTM(
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
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
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
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
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


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with attention mechanism"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize BiLSTM with Attention
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            pretrained_embeddings: Pretrained word embeddings
        """
        super(BiLSTMAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
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
    
    def attention_layer(self, lstm_output, lengths):
        """
        Apply attention mechanism
        
        Args:
            lstm_output: LSTM output [batch_size, seq_len, hidden_dim*2]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Context vector [batch_size, hidden_dim*2]
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)  # [batch_size, seq_len]
        
        # Create mask for padding
        batch_size, max_len = lstm_output.size(0), lstm_output.size(1)
        mask = torch.arange(max_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)
        
        # Apply mask (set padding positions to very negative number)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        
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
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply attention
        context, attention_weights = self.attention_layer(lstm_output, lengths)
        
        # Dropout
        context = self.dropout(context)
        
        # Fully connected layer
        output = self.fc(context)
        
        return output


class MultiTaskLSTM(nn.Module):
    """Multi-task LSTM for joint sentiment and emotion classification"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_sentiment_classes: int = 3,
                 num_emotions: int = 8,
                 dropout: float = 0.5,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize Multi-task LSTM
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            num_sentiment_classes: Number of sentiment classes
            num_emotions: Number of emotion labels
            dropout: Dropout probability
            pretrained_embeddings: Pretrained word embeddings
        """
        super(MultiTaskLSTM, self).__init__()
        
        # Shared layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific layers
        lstm_output_dim = hidden_dim * 2
        
        # Sentiment classification head
        self.sentiment_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sentiment_classes)
        )
        
        # Emotion detection head (multi-label)
        self.emotion_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions)
        )
    
    def forward(self, text, lengths):
        """
        Forward pass
        
        Args:
            text: Input text tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Tuple of (sentiment_logits, emotion_logits)
        """
        # Shared embedding
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pack and process through LSTM
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        # Task-specific outputs
        sentiment_output = self.sentiment_fc(hidden)
        emotion_output = self.emotion_fc(hidden)
        
        return sentiment_output, emotion_output


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM hybrid model for text classification"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 num_filters: int = 100,
                 filter_sizes: list = [3, 4, 5],
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize CNN-LSTM classifier
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters for each size
            filter_sizes: List of filter sizes
            hidden_dim: Dimension of LSTM hidden state
            num_classes: Number of output classes
            dropout: Dropout probability
            pretrained_embeddings: Pretrained word embeddings
        """
        super(CNNLSTMClassifier, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # LSTM layer
        cnn_output_dim = num_filters * len(filter_sizes)
        self.lstm = nn.LSTM(
            cnn_output_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, text, lengths):
        """
        Forward pass
        
        Args:
            text: Input text tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # CNN expects [batch_size, channels, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolution and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all conv outputs
        cnn_out = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        cnn_out = cnn_out.unsqueeze(1)  # [batch_size, 1, cnn_output_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_out)
        
        # Use last hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        # Fully connected
        output = self.fc(hidden)
        
        return output


if __name__ == "__main__":
    # Test models
    batch_size = 8
    seq_len = 50
    vocab_size = 10000
    
    # Generate random data
    text = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(10, seq_len, (batch_size,))
    
    print("Testing LSTM Classifier...")
    lstm_model = LSTMClassifier(vocab_size=vocab_size, num_classes=3)
    lstm_output = lstm_model(text, lengths)
    print(f"LSTM Output shape: {lstm_output.shape}")
    
    print("\nTesting BiLSTM with Attention...")
    bilstm_model = BiLSTMAttention(vocab_size=vocab_size, num_classes=3)
    bilstm_output = bilstm_model(text, lengths)
    print(f"BiLSTM+Attention Output shape: {bilstm_output.shape}")
    
    print("\nTesting Multi-task LSTM...")
    multitask_model = MultiTaskLSTM(vocab_size=vocab_size)
    sentiment_out, emotion_out = multitask_model(text, lengths)
    print(f"Sentiment Output shape: {sentiment_out.shape}")
    print(f"Emotion Output shape: {emotion_out.shape}")
    
    print("\n✅ All models initialized successfully!")
