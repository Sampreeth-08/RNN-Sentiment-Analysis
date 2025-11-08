import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the file paths
CSV_FILE_PATH = "results/metrics.csv"
PLOTS_DIR = "results/plots"

def plot_performance_vs_seq_length(df):
    """
    Generates and saves a plot of F1-score vs. Sequence Length
    for the best performing model configurations.
    """
    print(f"Generating plot for 'F1 vs. Sequence Length'...")
    
    # --- Filter for the most informative data ---
    # We want to compare the best models. Let's select:
    # 1. Optimizers: Adam and RMSProp (since SGD performed poorly)
    # 2. Clipping: Yes (since it generally improved performance)
    
    plot_df = df[
        (df['Optimizer'].isin(['Adam', 'RMSProp'])) &
        (df['Grad Clipping'] == 'Yes')
    ].copy()

    # Create a unique 'Model Configuration' label for the legend
    plot_df['Model Config'] = plot_df['Model'] + ' (' + plot_df['Activation'] + ')'
    plot_df.loc[plot_df['Model'].isin(['LSTM', 'BiLSTM']), 'Model Config'] = \
        plot_df['Model'] + ' (' + plot_df['Optimizer'] + ')'

    # --- Create the Plot ---
    plt.figure(figsize=(12, 8))
    
    # Get unique model configs to plot
    configs = plot_df['Model Config'].unique()
    
    for config in configs:
        model_data = plot_df[plot_df['Model Config'] == config]
        plt.plot(
            model_data['Seq Length'],
            model_data['F1'],
            label=config,
            marker='o'
        )

    plt.title('Model F1-Score vs. Sequence Length', fontsize=16)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Best F1-Score', fontsize=12)
    plt.xticks([25, 50, 100]) # Ensure we have ticks for each seq length
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    
    # --- Save the Plot ---
    save_path = os.path.join(PLOTS_DIR, "f1_vs_seq_length.png")
    plt.savefig(save_path)
    
    print(f"Plot saved to '{save_path}'")
    plt.close()

def plot_loss_vs_epoch(df):
    """
    Generates and saves a plot of Training and Validation Loss vs. Epochs
    for the 'Best Model' and 'Worst Model'.
    """
    print(f"Generating plot for 'Loss vs. Epochs'...")
    
    # --- Create the Plot ---
    # We create two subplots: one for the Best Model, one for the Worst.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Training and Validation Loss vs. Epochs', fontsize=16, y=1.02)
    
    # --- Plot 1: Best Model ---
    best_df = df[df['experiment'] == 'Best Model (BiLSTM-RMSProp)']
    ax1.plot(best_df['epoch'], best_df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(best_df['epoch'], best_df['val_loss'], label='Validation Loss', marker='x')
    ax1.set_title('Best Model (BiLSTM-RMSProp)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCEWithLogitsLoss')
    ax1.legend()
    ax1.grid(True, linestyle='--', linewidth=0.5)
    
    # --- Plot 2: Worst Model ---
    worst_df = df[df['experiment'] == 'Worst Model (LSTM-SGD)']
    ax2.plot(worst_df['epoch'], worst_df['train_loss'], label='Train Loss', marker='o')
    ax2.plot(worst_df['epoch'], worst_df['val_loss'], label='Validation Loss', marker='x')
    ax2.set_title('Worst Model (LSTM-SGD)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BCEWithLogitsLoss')
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # --- Save the Plot ---
    save_path = os.path.join(PLOTS_DIR, "loss_vs_epochs.png")
    plt.savefig(save_path)
    
    print(f"Plot saved to '{save_path}'")
    plt.close()

if __name__ == "__main__":
    # Create the plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("--- Running Plotting Script ---")
    
    # --- Generate Plot 1 ---
    try:
        metrics_df = pd.read_csv(CSV_FILE_PATH)
        plot_performance_vs_seq_length(metrics_df)
    except FileNotFoundError:
        print(f"Error: Could not find results file at '{CSV_FILE_PATH}'")
        print("Please run 'python -m src.evaluate' first.")
    except Exception as e:
        print(f"An error occurred during 'F1 vs. Seq Length' plotting: {e}")

    # --- Generate Plot 2 ---
    try:
        epoch_loss_df = pd.read_csv("results/epoch_loss.csv")
        plot_loss_vs_epoch(epoch_loss_df)
    except FileNotFoundError:
        print(f"\nError: Could not find results file at 'results/epoch_loss.csv'")
        print("Please run 'python -m src.log_epochs' first.")
    except Exception as e:
        print(f"An error occurred during 'Loss vs. Epochs' plotting: {e}")

    print("\nPlotting script complete.")