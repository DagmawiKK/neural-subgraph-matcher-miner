import os
import random
import numpy as np
import torch

def set_all_seeds(seed):
    """Set seeds for all random number generators to ensure reproducibility."""
    print(f"Setting all random seeds to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Set deterministic algorithms (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to {seed}")