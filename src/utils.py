import torch
import random
import numpy as np

def set_seeds(seed_value=42):
    """
    Sets the random seeds for reproducibility as required by the project.
    [cite: 41, 42, 43, 44, 45]
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # These are needed for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False