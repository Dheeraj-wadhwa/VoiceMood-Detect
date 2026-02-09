import os
import librosa
import numpy as np
import pickle
from tqdm import tqdm
try:
    import config
    from utils import ensure_dir, parse_emotion_from_filename
except ImportError:
    from . import config
    from .utils import ensure_dir, parse_emotion_from_filename

def load_audio(file_path):
    """Loads and resamples audio."""
    try:
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def pad_truncate(y, target_length):
    """Pads or truncates audio to a fixed length."""
    if len(y) > target_length:
        return y[:target_length]
    else:
        padding = target_length - len(y)
        return np.pad(y, (0, padding), mode='constant')

def extract_features(y, sr):
    """Extracts MFCC features."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC, 
                                n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    # MFCC shape: (n_mfcc, time_steps)
    return mfcc

def process_dataset():
    ensure_dir(config.PROCESSED_DATA_DIR)
    
    if not os.path.exists(config.RAW_DATA_DIR):
        print(f"ERROR: Raw data directory not found at {config.RAW_DATA_DIR}")
        print("Please download the SAVEE dataset and place it there.")
        return

    data = []
    labels = []
    
    # Walk through the directory
    files = []
    for root, _, filenames in os.walk(config.RAW_DATA_DIR):
        for f in filenames:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} audio files.")
    
    for file_path in tqdm(files, desc="Processing Audio"):
        filename = os.path.basename(file_path)
        emotion_code = parse_emotion_from_filename(filename)
        
        if emotion_code not in config.EMOTIONS:
            continue
            
        emotion_label = config.EMOTIONS[emotion_code]
        label_idx = config.CLASS_TO_IDX[emotion_label]
        
        y, sr = load_audio(file_path)
        if y is None:
            continue
            
        # Clean silence (optional but recommended)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Pad/Truncate
        y_fixed = pad_truncate(y, config.SAMPLES_PER_TRACK)
        
        # Extract features
        mfcc = extract_features(y_fixed, sr)
        
        data.append(mfcc)
        labels.append(label_idx)
        
    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)
    
    print(f"Processed Data Shape: {X.shape}")
    print(f"Labels Shape: {y.shape}")
    
    # Save processed data
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "X.npy"), X)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "y.npy"), y)
    print("Data processing complete. Saved to processed/")

if __name__ == "__main__":
    process_dataset()
