import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils import set_seeds # Import our seed function
from torch.utils.data import DataLoader, TensorDataset
import torch

DATA_FILE_PATH = "data/IMDB Dataset.csv"

def clean_text(text):
    """
    Cleans text data by lowercasing, removing HTML tags, 
    and removing punctuation/special characters.
    """
    text = text.lower() # Lowercase all text 
    text = re.sub(r'<[^>]+>', ' ', text) # Remove HTML tags (common in this dataset)
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and special characters 
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def load_and_preprocess_data(vocab_size=10000, max_length=50):
    """
    Loads the local CSV, preprocesses, and pads the IMDb movie review dataset.
    
    Args:
        vocab_size (int): The number of most frequent words to keep. 
        max_length (int): The fixed sequence length to pad/truncate to. 
    
    Returns:
        (tuple): (x_train, y_train), (x_test, y_test), tokenizer
                 Padded sequences, their labels, and the fitted tokenizer.
    """
    # Ensure reproducibility right before loading/splitting
    set_seeds()
    
    # --- 1. Load Data from local CSV ---
    print(f"Loading dataset from '{DATA_FILE_PATH}'...")
    try:
        # The file was saved with a non-standard encoding
        df = pd.read_csv(DATA_FILE_PATH, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: File not found at '{DATA_FILE_PATH}'")
        print("Please make sure 'IMDB Dataset.csv' is inside the 'data/' directory.")
        return None, None, None
    
    print(f"Loaded {len(df)} total reviews.")
    
    # --- 2. Preprocess Text and Labels ---
    print("Cleaning text data...")
    df['review_cleaned'] = df['review'].apply(clean_text)
    
    # Convert labels: positive -> 1, negative -> 0
    df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    reviews = df['review_cleaned'].values
    labels = df['sentiment_label'].values

    # --- 3. Create 50/50 Train/Test Split ---
    # Requirement: "Use the predefined 50/50 split" 
    print("Creating 50/50 train/test split...")
    x_train, x_test, y_train, y_test = train_test_split(
        reviews, 
        labels, 
        test_size=0.5, 
        random_state=42, # for reproducibility
        stratify=labels # ensure balanced split
    )
    
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    # --- 4. Tokenize and Convert to Sequences ---
    print(f"Tokenizing and limiting vocabulary to {vocab_size} words...")
    # Keep top 'vocab_size' words, use <OOV> for out-of-vocabulary words
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train) # Build vocab from training data
    
    # Convert each review to a sequence of token IDs 
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    # --- 5. Pad Sequences ---
    print(f"Padding/truncating sequences to length {max_length}...")
    x_train_padded = pad_sequences(x_train_seq, maxlen=max_length, padding='post', truncating='post')
    x_test_padded = pad_sequences(x_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    # Ensure labels are numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)
            
    return (x_train_padded, y_train), (x_test_padded, y_test), tokenizer

def create_dataloaders(x_train, y_train, x_test, y_test, batch_size=32):
    """
    Converts numpy arrays into PyTorch DataLoaders.
    
    Args:
        x_train (np.array): Training features
        y_train (np.array): Training labels
        x_test (np.array): Test features
        y_test (np.array): Test labels
        batch_size (int): Batch size for the DataLoaders
        
    Returns:
        (tuple): (train_loader, test_loader)
    """
    # Convert data to PyTorch Tensors
    # We need LongTensor for token IDs and FloatTensor for labels
    train_data = TensorDataset(torch.from_numpy(x_train).long(), 
                               torch.from_numpy(y_train).float())
    test_data = TensorDataset(torch.from_numpy(x_test).long(), 
                              torch.from_numpy(y_test).float())
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    return train_loader, test_loader

# --- Test block ---
# if __name__ == "__main__":
#     print("--- Testing Preprocessing Script ---")
#     # Test with one of the required sequence lengths
#     (x_train, y_train), (x_test, y_test), tok = load_and_preprocess_data(
#         vocab_size=10000, 
#         max_length=50
#     )
    
#     if x_train is not None:
#         print("\n--- Results ---")
#         print(f"x_train shape: {x_train.shape}")
#         print(f"y_train shape: {y_train.shape}")
#         print(f"x_test shape: {x_test.shape}")
#         print(f"y_test shape: {y_test.shape}")
#         print("\nExample processed review (token IDs):")
#         print(x_train[0])
#         print(f"Example label: {y_train[0]}")
        
#         # Check vocab
#         word_count = len(tok.word_index)
#         print(f"Total words in tokenizer vocabulary: {word_count}")
#         print("Example word-to-index mapping:")
#         print(f"'the': {tok.word_index.get('the')}")
#         print(f"'movie': {tok.word_index.get('movie')}")

#         print("\nPreprocessing test complete.")

# --- Test block ---
if __name__ == "__main__":
    print("--- Testing Preprocessing Script ---")
    BATCH_SIZE = 32
    
    # 1. Load and preprocess
    (x_train, y_train), (x_test, y_test), tok = load_and_preprocess_data(
        vocab_size=10000, 
        max_length=50
    )
    
    if x_train is not None:
        print("\n--- Testing DataLoader Creation ---")
        
        # 2. Create DataLoaders
        train_loader, test_loader = create_dataloaders(
            x_train, y_train, x_test, y_test, 
            batch_size=BATCH_SIZE
        )
        
        print(f"Created train_loader and test_loader with batch size {BATCH_SIZE}.")
        
        # 3. Check one batch
        data_iter = iter(train_loader)
        features, labels = next(data_iter)
        
        print("\n--- Results ---")
        print(f"Shape of one feature batch (should be [Batch, SeqLen]): {features.shape}")
        print(f"Shape of one label batch (should be [Batch]): {labels.shape}")
        
        assert features.shape == (BATCH_SIZE, 50)
        assert labels.shape == (BATCH_SIZE,)
        
        print("\nPreprocessing test complete.")