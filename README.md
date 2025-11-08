# Comparative Analysis of RNN Architectures for Sentiment Classification

This project implements and systematically evaluates multiple Recurrent Neural Network (RNN) architectures for binary sentiment classification on the IMDb movie review dataset.

## Project Structure

```
.
├── data/
│   └── IMDB Dataset.csv      # (Must be downloaded manually)
├── results/
│   ├── metrics.csv           # Full results of all 72 experiments
│   ├── epoch_loss.csv        # Detailed epoch logs for best/worst models
│   └── plots/                # Generated plots for the report
├── src/
│   ├── utils.py              # Utility functions (seeding for reproducibility)
│   ├── preprocess.py         # Data loading, cleaning, and tokenization
│   ├── models.py             # PyTorch model definitions (RNN, LSTM, BiLSTM)
│   ├── train.py              # Training and evaluation loops
│   ├── evaluate.py           # Main script to run full experimental suite
│   ├── log_epochs.py         # Script to log detailed epoch data
│   └── plot.py               # Script to generate all report plots
├── README.md
└── report.pdf                # Final project report
```

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install torch numpy pandas scikit-learn matplotlib tensorflow
```

*Note: `tensorflow` is only used for its robust text preprocessing tools (`Tokenizer`, `pad_sequences`). The models themselves are built entirely in PyTorch.*

### 3. Download Data

Download the IMDb dataset from Kaggle:
[IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Place the `IMDB Dataset.csv` file inside the `data/` directory:

```
data/IMDB Dataset.csv
```

## Hardware & Environment

The experiments were run on the following hardware, which significantly impacts training time:

  * **PyTorch version:** 2.5.1+cu121
  * **CUDA available:** True
  * **CUDA version:** 12.1
  * **GPU:** NVIDIA RTX A6000

## Running the Project

All scripts should be run from the root project directory using the `python -m` command.

### Expected Runtime and Output Files

  * **`python -m src.evaluate`**:

      * **Runtime**: This is the main experiment script. On the hardware specified above, each of the 72 experiments took **1.5 - 2.7 seconds** to run (for 5 epochs each). The total runtime was approximately **11-12 minutes**. On a standard CPU, this will take significantly longer (potentially 1+ hours).
      * **Output**: Generates `results/metrics.csv`.

  * **`python -m src.log_epochs`**:

      * **Runtime**: This script trains two models for 10 epochs each (approx. 1-2 minutes on the specified GPU).
      * **Output**: Generates `results/epoch_loss.csv`.

  * **`python -m src.plot`**:

      * **Runtime**: This script runs very quickly (approx. 10-20 seconds).
      * **Output**: Generates `results/plots/f1_vs_seq_length.png` and `results/plots/loss_vs_epochs.png`.

-----

### 1. Run Full Evaluation (Reproduce Experiments)

To run all 72 experimental configurations and generate the `results/metrics.csv` file:

```bash
python -m src.evaluate
```

### 2. (Optional) Log Detailed Epoch Data

To re-train the best and worst models and log their epoch-by-epoch loss for the second plot:

```bash
python -m src.log_epochs
```

### 3. Generate Plots

To generate the plots used in the report based on the results files:

```bash
python -m src.plot
```

## Key Findings

  * **Optimal Configuration:** Bidirectional LSTM with RMSProp optimizer, sequence length of 100, and gradient clipping enabled (F1-Score: 0.8218).
  * **Sequence Length:** Strongly correlated with performance; longer sequences (100 words) consistently yielded better results than shorter ones (25, 50).
  * **Optimizers:** Adam and RMSProp were highly effective, while SGD failed to learn the task.
  * **Gradient Clipping:** Critical for stabilizing standard RNNs and provided a minor but consistent performance boost for LSTM-based models.
