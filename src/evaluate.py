import os
import time
import torch
import torch.nn as nn
import pandas as pd
from itertools import product

# Import all our custom modules
from src.utils import set_seeds
from src.models import SentimentRNN
from src.preprocess import load_and_preprocess_data, create_dataloaders
from src.train import get_optimizer, train_epoch, evaluate_epoch

# --- 1. Define All Experimental Configurations ---

FIXED_PARAMS = {
    'vocab_size': 10000,
    'embedding_dim': 100,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.4, 
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 5 
}

# --- THIS IS THE FIX ---
# Changed "ReLU" to "relu" and "Tanh" to "tanh"
VAR_PARAMS = {
    'model_type': ["RNN", "LSTM", "BiLSTM"],
    'activation': ["relu", "tanh"],
    'optimizer_name': ["Adam", "SGD", "RMSProp"],
    'seq_length': [25, 50, 100],
    'use_grad_clipping': [False, True]
}
# -------------------------

# --- 2. Main Experiment Function ---

def run_experiment(config):
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    seq_len = config['seq_length']
    batch_size = config['batch_size']
    vocab_size = config['vocab_size']
    
    (x_train, y_train), (x_test, y_test), _ = load_and_preprocess_data(
        vocab_size=vocab_size, 
        max_length=seq_len
    )
    
    train_loader, test_loader = create_dataloaders(
        x_train, y_train, x_test, y_test, 
        batch_size=batch_size
    )

    model = SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        model_type=config['model_type'],
        dropout=config['dropout'],
        activation=config['activation']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, config['optimizer_name'], config['learning_rate'])
    
    best_val_f1 = -1
    best_val_acc = -1 # Initialize best_val_acc
    total_epoch_time = 0

    for epoch in range(1, config['num_epochs'] + 1):
        
        train_loss, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            use_grad_clipping=config['use_grad_clipping']
        )
        
        val_loss, val_acc, val_f1 = evaluate_epoch(
            model, test_loader, criterion, device
        )
        
        total_epoch_time += epoch_time
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            
    avg_epoch_time = total_epoch_time / config['num_epochs']
    
    result = {
        "Model": config['model_type'],
        "Activation": config['activation'] if config['model_type'] == "RNN" else "N/A",
        "Optimizer": config['optimizer_name'],
        "Seq Length": config['seq_length'],
        "Grad Clipping": "Yes" if config['use_grad_clipping'] else "No",
        "Accuracy": round(best_val_acc, 4),
        "F1": round(best_val_f1, 4),
        "Epoch Time (s)": round(avg_epoch_time, 2)
    }
    
    return result

# --- 3. Main Execution Block ---

if __name__ == "__main__":
    print("--- Starting Full Experiment Evaluation ---")
    
    os.makedirs("results", exist_ok=True)
    
    all_experiments = []
    
    for seq_len in VAR_PARAMS['seq_length']:
        for opt in VAR_PARAMS['optimizer_name']:
            for clip in VAR_PARAMS['use_grad_clipping']:
                for act in VAR_PARAMS['activation']:
                    exp_config = FIXED_PARAMS.copy()
                    exp_config.update({
                        'model_type': 'RNN',
                        'activation': act,
                        'optimizer_name': opt,
                        'seq_length': seq_len,
                        'use_grad_clipping': clip
                    })
                    all_experiments.append(exp_config)
                
                for model in ['LSTM', 'BiLSTM']:
                    exp_config = FIXED_PARAMS.copy()
                    exp_config.update({
                        'model_type': model,
                        'activation': 'N/A',
                        'optimizer_name': opt,
                        'seq_length': seq_len,
                        'use_grad_clipping': clip
                    })
                    all_experiments.append(exp_config)

    # Note: The total number of experiments is still 72
    # 3 seq_len * 3 optimizers * 2 clipping * (2 RNN activations + 2 LSTM/BiLSTM) = 72
    print(f"Total number of experiments to run: {len(all_experiments)}") 
    
    all_results = []
    
    for i, config in enumerate(all_experiments):
        print(f"\n--- Running Experiment {i+1}/{len(all_experiments)} ---")
        print(f"Config: {config['model_type']} | {config['activation']} | {config['optimizer_name']} | "
              f"Seq: {config['seq_length']} | Clip: {config['use_grad_clipping']}")
        
        try:
            result = run_experiment(config)
            all_results.append(result)
            print(f"Result: Acc={result['Accuracy']}, F1={result['F1']}, Time={result['Epoch Time (s)']}")
        except Exception as e:
            print(f"!!! Experiment FAILED: {e}")
            all_results.append({
                "Model": config['model_type'],
                "Activation": config['activation'],
                "Optimizer": config['optimizer_name'],
                "Seq Length": config['seq_length'],
                "Grad Clipping": "Yes" if config['use_grad_clipping'] else "No",
                "Accuracy": "FAIL", "F1": "FAIL", "Epoch Time (s)": "FAIL"
            })

    print("\n--- All Experiments Complete ---")
    
    results_df = pd.DataFrame(all_results)
    
    columns_order = ["Model", "Activation", "Optimizer", "Seq Length", 
                     "Grad Clipping", "Accuracy", "F1", "Epoch Time (s)"]
    results_df = results_df[columns_order]
    
    output_path = "results/metrics.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print("\nFinal Results Table:")
    print(results_df.to_string())