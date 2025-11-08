import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

# Import our custom modules
from src.utils import set_seeds
from src.models import SentimentRNN
from src.preprocess import load_and_preprocess_data, create_dataloaders

def get_optimizer(model, optimizer_name, learning_rate=0.001):
    """
    Returns a PyTorch optimizer based on its name.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_epoch(model, dataloader, optimizer, criterion, device, use_grad_clipping=False):
    """
    Runs one full epoch of training.
    """
    model.train()  # Set model to training mode
    total_loss = 0
    start_time = time.time()

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping 
        if use_grad_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimize
        optimizer.step()
        
        total_loss += loss.item()
        
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    return avg_loss, epoch_time

def evaluate_epoch(model, dataloader, criterion, device):
    """
    Runs one full epoch of evaluation.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculations
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            # Apply sigmoid and round to get binary predictions (0 or 1)
            preds = torch.round(torch.sigmoid(outputs))
            
            # Store preds and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    # Calculate metrics [cite: 31, 32]
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1

# --- Test block ---
if __name__ == "__main__":
    print("--- Testing Training Script ---")
    
    # --- 1. Set Hyperparameters for one test run ---
    # Data params
    VOCAB_SIZE = 10000    # [cite: 15]
    SEQ_LENGTH = 50       # 
    
    # Model params [cite: 21, 23, 24, 25]
    MODEL_TYPE = "lstm"
    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.4
    ACTIVATION = "tanh" # Only used for 'rnn' type
    
    # Training params [cite: 21, 26]
    OPTIMIZER = "adam"
    LEARNING_RATE = 0.001
    USE_GRAD_CLIPPING = True
    BATCH_SIZE = 32
    NUM_EPOCHS = 3 # Just for a quick test

    # --- 2. Setup (Seed, Device, Data) ---
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test), tokenizer = load_and_preprocess_data(
        vocab_size=VOCAB_SIZE, 
        max_length=SEQ_LENGTH
    )
    
    train_loader, test_loader = create_dataloaders(
        x_train, y_train, x_test, y_test, 
        batch_size=BATCH_SIZE
    )
    print("DataLoaders created.")

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = SentimentRNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        model_type=MODEL_TYPE,
        dropout=DROPOUT,
        activation=ACTIVATION
    ).to(device)
    
    # Use BCEWithLogitsLoss for binary classification [cite: 27]
    # It combines a Sigmoid layer and BCELoss for better numerical stability
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = get_optimizer(model, OPTIMIZER, LEARNING_RATE)
    
    print(f"Model, Criterion, and Optimizer initialized for '{MODEL_TYPE}' test.")
    
    # --- 4. Run Training Loop ---
    print("\n--- Starting Test Training Loop ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        
        train_loss, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device, USE_GRAD_CLIPPING
        )
        
        val_loss, val_acc, val_f1 = evaluate_epoch(
            model, test_loader, criterion, device
        )
        
        print(f"Epoch: {epoch}/{NUM_EPOCHS}")
        print(f"\tTrain Loss: {train_loss:.4f} | Epoch Time: {epoch_time:.2f}s")
        print(f"\tVal. Loss: {val_loss:.4f} | Val. Acc: {val_acc:.4f} | Val. F1: {val_f1:.4f}")

    print("\nTraining script test complete.")