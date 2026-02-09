import torch
from torch.utils.data import Dataset
import numpy as np

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Resulting shape: (C, H, W) or (C, L)
        # MFCC X is (N, n_mfcc, time)
        # We need to make sure it matches what the model expects.
        # CNN usually is (Batch, C, n_mfcc, time) for 2D or (Batch, C, time) for 1D.
        # Our model will likely treat it as an image (1 Channel, n_mfcc, time)
        
        features = self.X[idx]
        label = self.y[idx]
        
        # Add channel dimension: (1, n_mfcc, time)
        features = features.unsqueeze(0)
        
        return features, label
