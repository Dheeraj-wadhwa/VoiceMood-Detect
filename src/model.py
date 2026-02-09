import torch
import torch.nn as nn
try:
    import config
except ImportError:
    from . import config

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNLSTMModel, self).__init__()
        
        # CNN Block
        self.cnn = nn.Sequential(
            # Input: (1, n_mfcc, time) -> (1, 13, 93) approx
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Compute the output size of CNN to feed into LSTM
        # We can do a dry run or calculate.
        # Height: 13 -> 6 -> 3 -> 1
        # Width: Time dimension reduces by factor of 8 (2*2*2)
        
        self.lstm_input_size = 64 * 1  # 64 channels * 1 remaining height feature
        self.hidden_dim = 128
        self.num_layers = 1
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64), # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (Batch, 1, n_mfcc, time)
        
        batch_size = x.size(0)
        
        x = self.cnn(x) 
        # Output: (Batch, 64, H_new, W_new)
        # We want to treat W_new as the time sequence for LSTM.
        # Permute to (Batch, W_new, 64 * H_new)
        
        x = x.permute(0, 3, 1, 2) # (Batch, W_new, 64, H_new)
        x = x.reshape(batch_size, x.size(1), -1) # (Batch, SequenceLength, Features)
        
        output, (hn, cn) = self.lstm(x)
        
        # Use the last hidden state? Or pool?
        # output is (Batch, SeqLen, HiddenDim*2)
        # We can take the last time step
        
        out = output[:, -1, :] 
        
        out = self.fc(out)
        return out
