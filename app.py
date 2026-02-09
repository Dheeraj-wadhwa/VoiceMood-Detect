import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
from src import config
from src.predict import EmotionPredictor

# Page Config
st.set_page_config(
    page_title="VoiceMood Detect",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Title and Sidebar
st.title("🎙️ VoiceMood Detect")
st.markdown("### *Artificial Intelligence for Emotion Recognition*")
st.sidebar.image("https://img.icons8.com/color/480/waveform.png", width=150)
st.sidebar.markdown("---")
st.sidebar.header("Settings")
model_source = st.sidebar.selectbox("Model Version", ["Best Model (v1)", "Generic (Baseline)"])

# Initialize Predictor
@st.cache_resource
def get_predictor():
    return EmotionPredictor()

predictor = get_predictor()

# Main Area
col1, col2 = st.columns([1, 2])

with col1:
    st.info("Upload an audio file or record your voice to analyze the underlying emotions.")
    uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
    
    # Placeholder for Audio Recorder (Streamlit currently doesn't support native mic recording easily without components, 
    # but we will assume file upload for MVP or use stash custom component usually)
    st.markdown("---")
    st.warning("Ensure audio is clear and 3-5 seconds long for best results.")

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    with col2:
        st.subheader("Trace Analysis")
        
        # Plot Waveform
        y, sr = librosa.load(tmp_path, sr=config.SAMPLE_RATE)
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#ff4b4b")
        ax.set_title("Waveform")
        ax.set_axis_off()
        st.pyplot(fig)
        
        # Audio Player
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Emotion 🔍"):
            with st.spinner("Processing signal... extracting MFCCs... running inference..."):
                predicted_class, probs = predictor.predict(tmp_path)
                
            st.success("Analysis Complete!")
            
            # Display Results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>Detected Emotion</h2>
                    <h1 style="color: #4CAF50;">{predicted_class.upper()}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            with result_col2:
                # Probability Bar Chart
                st.bar_chart(dict(zip(config.EMOTION_LIST, probs)))
                
            # Clean up
            os.remove(tmp_path)

else:
    with col2:
        st.markdown("""
        ### How it works?
        1. **Preprocessing**: The audio is resampled to 16kHz and silence is trimmed.
        2. **Feature Extraction**: MFCCs (Mel-Frequency Cepstral Coefficients) are computed.
        3. **Deep Learning**: A Hybrid CNN-LSTM network processes the spectral and temporal features.
        4. **Prediction**: The model outputs the probability for each emotion.
        """)
