import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 model_type="lstm", dropout=0.5, activation='tanh'):
        
        super(SentimentRNN, self).__init__()
        
        self.model_type = model_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        
        # --- THIS IS THE FIRST FIX ---
        # Check for 'bilstm' correctly
        is_bidirectional = self.model_type == "bilstm"
        # -----------------------------
        
        if self.model_type == "lstm" or self.model_type == "bilstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=is_bidirectional
            )
        elif self.model_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=activation,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        else:
            # This error will now catch the mismatch if 'BiLSTM' wasn't fixed
            raise ValueError(f"Unsupported model_type: '{self.model_type}'")
            
        self.dropout = nn.Dropout(dropout)
        
        fc_input_features = hidden_size * 2 if is_bidirectional else hidden_size
        
        self.fc = nn.Linear(fc_input_features, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        
        # --- THIS IS THE SECOND (CRITICAL) FIX ---
        # This solves the UnboundLocalError for the RNN model
        
        if self.model_type == "lstm" or self.model_type == "bilstm":
            output, (hidden, cell) = self.rnn(embedded)
        else: # 'rnn'
            # We must assign to 'hidden' so the next block can find it!
            output, hidden = self.rnn(embedded)
        # -------------------------------------------

        if self.model_type == "bilstm":
            # hidden is (num_layers * 2, batch_size, hidden_size)
            # Concat the last forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # hidden is (num_layers, batch_size, hidden_size)
            # Just take the last layer's hidden state
            hidden = hidden[-1,:,:]
            
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output.squeeze()

# --- Test block (unchanged) ---
if __name__ == "__main__":
    print("--- Testing Model Class ---")
    
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.4
    BATCH_SIZE = 32
    SEQ_LENGTH = 50
    
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    
    print("\nTesting LSTM model...")
    model_lstm = SentimentRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS,
                              model_type="lstm", dropout=DROPOUT)
    output_lstm = model_lstm(dummy_input)
    print(f"LSTM output shape (should be {BATCH_SIZE}): {output_lstm.shape}")
    assert output_lstm.shape == (BATCH_SIZE,)

    print("\nTesting BiLSTM model...")
    # Test with the new 'bilstm' string
    model_bilstm = SentimentRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS,
                                model_type="bilstm", dropout=DROPOUT)
    output_bilstm = model_bilstm(dummy_input)
    print(f"BiLSTM output shape (should be {BATCH_SIZE}): {output_bilstm.shape}")
    assert output_bilstm.shape == (BATCH_SIZE,)

    print("\nTesting RNN (ReLU) model...")
    model_rnn = SentimentRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS,
                             model_type="rnn", dropout=DROPOUT, activation='relu')
    output_rnn = model_rnn(dummy_input)
    print(f"RNN (ReLU) output shape (should be {BATCH_SIZE}): {output_rnn.shape}")
    assert output_rnn.shape == (BATCH_SIZE,)
    
    print("\nModel class test complete.")