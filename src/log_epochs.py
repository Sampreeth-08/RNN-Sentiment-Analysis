import os
import torch
import torch.nn as nn
import pandas as pd

# Import all our custom modules
from src.utils import set_seeds
from src.models import SentimentRNN
from src.preprocess import load_and_preprocess_data, create_dataloaders
from src.train import get_optimizer, train_epoch, evaluate_epoch

def run_and_log_epochs(config, experiment_label):
    """
    Runs a single experiment and logs the train/val loss for every epoch.
    """
    print(f"--- Running: {experiment_label} ---")
    
    # --- A. Setup (Seed, Device, Data) ---
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    (x_train, y_train), (x_test, y_test), _ = load_and_preprocess_data(
        vocab_size=config['vocab_size'], 
        max_length=config['seq_length']
    )
    
    train_loader, test_loader = create_dataloaders(
        x_train, y_train, x_test, y_test, 
        batch_size=config['batch_size']
    )

    # --- B. Initialize Model, Loss, and Optimizer ---
    model = SentimentRNN(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        model_type=config['model_type'],
        dropout=config['dropout'],
        activation=config['activation']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, config['optimizer_name'], config['learning_rate'])
    
    # --- C. Run Training Loop and Log Epochs ---
    epoch_logs = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        
        train_loss, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            use_grad_clipping=config['use_grad_clipping']
        )
        
        val_loss, val_acc, val_f1 = evaluate_epoch(
            model, test_loader, criterion, device
        )
        
        print(f"Epoch {epoch}/{config['num_epochs']} | Train Loss: {train_loss:.4f} | Val. Loss: {val_loss:.4f}")
        
        # Log the results for this epoch
        epoch_logs.append({
            'experiment': experiment_label,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
            
    return epoch_logs

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Define the two configurations we need to run ---
    
    # These params are fixed
    BASE_PARAMS = {
        'vocab_size': 10000,
        'embedding_dim': 100,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.4,
        'batch_size': 32,
        'num_epochs': 10 # Run for 10 epochs to see the curve
    }

    # Configuration for the BEST model
    BEST_CONFIG = BASE_PARAMS.copy()
    BEST_CONFIG.update({
        'model_type': 'BiLSTM',
        'activation': 'N/A',
        'optimizer_name': 'RMSProp',
        'seq_length': 100,
        'use_grad_clipping': True,
        'learning_rate': 0.001
    })
    
    # Configuration for the WORST model
    WORST_CONFIG = BASE_PARAMS.copy()
    WORST_CONFIG.update({
        'model_type': 'LSTM',
        'activation': 'N/A',
        'optimizer_name': 'SGD',
        'seq_length': 25,
        'use_grad_clipping': False,
        'learning_rate': 0.001 
    })
    
    all_epoch_logs = []
    
    # --- Run both experiments ---
    all_epoch_logs.extend(run_and_log_epochs(BEST_CONFIG, "Best Model (BiLSTM-RMSProp)"))
    all_epoch_logs.extend(run_and_log_epochs(WORST_CONFIG, "Worst Model (LSTM-SGD)"))
    
    # --- Save results to CSV ---
    results_df = pd.DataFrame(all_epoch_logs)
    output_path = "results/epoch_loss.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nEpoch loss data saved to {output_path}")