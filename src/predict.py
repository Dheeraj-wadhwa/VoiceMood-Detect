import torch
import numpy as np
import librosa
import os
try:
    import config
    from model import CNNLSTMModel
except ImportError:
    from . import config
    from .model import CNNLSTMModel

class EmotionPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTMModel(num_classes=len(config.EMOTIONS))
        
        if model_path is None:
            # Try to find best model
            model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print("Warning: No model found. Predictions will be random initialized.")
            
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_audio(self, file_path):
        # reuse logic roughly or import
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        # trim
        y, _ = librosa.effects.trim(y, top_db=20)
        # pad
        if len(y) > config.SAMPLES_PER_TRACK:
            y = y[:config.SAMPLES_PER_TRACK]
        else:
            padding = config.SAMPLES_PER_TRACK - len(y)
            y = np.pad(y, (0, padding), mode='constant')
            
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC, 
                                    n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        # Add batch and channel dims: (1, 1, n_mfcc, time)
        input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return input_tensor
        
    def predict(self, file_path):
        input_tensor = self.preprocess_audio(file_path).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            
        probs_np = probs.cpu().numpy()[0]
        prediction = np.argmax(probs_np)
        predicted_class = config.IDX_TO_CLASS[prediction]
        
        return predicted_class, probs_np
