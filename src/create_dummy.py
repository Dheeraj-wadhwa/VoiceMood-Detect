import numpy as np
import scipy.io.wavfile as wav
import os
import config

def create_dummy():
    # Ensure directory exists
    if not os.path.exists(config.RAW_DATA_DIR):
        os.makedirs(config.RAW_DATA_DIR)
        
    # Generate a simple sine wave
    sr = 16000
    t = np.linspace(0, 3, sr * 3)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Save a few dummy files covering different emotions
    filenames = ["DC_a01.wav", "DC_d01.wav", "JE_f01.wav", "JK_h01.wav", "KL_n01.wav", "DC_sa01.wav", "DC_su01.wav"]
    
    for f in filenames:
        path = os.path.join(config.RAW_DATA_DIR, f)
        wav.write(path, sr, y.astype(np.float32))
        print(f"Created dummy file: {path}")

if __name__ == "__main__":
    create_dummy()
