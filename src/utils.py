import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_waveform(y, sr, title="Waveform"):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr, title="Spectrogram"):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def parse_emotion_from_filename(filename):
    """
    Parses SAVEE filename to extract emotion code.
    Format example: DC_a01.wav -> 'a' (anger)
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Split by underscore. Example: DC_a01 or JE_a01
    parts = name.split('_')
    if len(parts) < 2:
        return None
    
    code_part = parts[1]
    # The code part is like 'a01', 'sa01'. We need the letters.
    # Regex is safest, or just simple parsing
    import re
    match = re.match(r"([a-z]+)", code_part)
    if match:
        return match.group(1)
    return None
