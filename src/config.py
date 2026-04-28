import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "SAVEE")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Audio Config
SAMPLE_RATE = 16000
DURATION = 3.0  # Seconds to pad/truncate to
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Feature Extraction (MFCC)(Mel-Frequency Cepstral Coefficients)
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Training
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Classes
EMOTIONS = {
    'a': 'anger',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happiness',
    'n': 'neutral',
    'sa': 'sadness',
    'su': 'surprise'
}

EMOTION_LIST = list(EMOTIONS.values())
CLASS_TO_IDX = {e: i for i, e in enumerate(EMOTION_LIST)}
IDX_TO_CLASS = {i: e for i, e in enumerate(EMOTION_LIST)}
