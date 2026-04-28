# VoiceMood Detect 🎙️🎭

## Overview
VoiceMood Detect is a deep learning-based application designed to recognize emotions from speech audio. It utilizes a hybrid CNN-LSTM architecture trained on the SAVEE dataset to classify emotions such as Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.

## features
- **Emotion Recognition**: Detects 7 distinct emotions from speech.
- **Deep Learning Model**: Uses MFCC features fed into a CNN-LSTM network.
- **Interactive UI**: Built with Streamlit for real-time recording and analysis.
- **Visualizations**: Dynamic waveform and emotion probability charts.

## Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Data Setup**:
    - Place the SAVEE dataset audio files in `data/SAVEE/`.
    - Ensure filenames follow the format `DC_[emotion][number].wav`, `JE_[emotion][number].wav`, etc., or similar standard SAVEE naming.

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
## PIPELINE SUMMARY 
```bash
1. `config.py` & `utils.py`: Setup and global definitions.
2. `preprocess.py`**: Transformation of raw audio into mathematical representations.
3. `dataset.py` & `model.py`: Structure definitions for data feeding and network layout.
4. `train.py`: Execution of optimization, generating the `.pth` weights.
5. `predict.py`: Inference logic to utilize trained weights.
6. `app.py`: End-user interaction and visual results.
```

## Project Structure
- `data/`: Dataset storage.
- `models/`: Saved model checkpoints.
- `src/`: Source code for preprocessing, training, and inference.
- `app.py`: Streamlit frontend.
