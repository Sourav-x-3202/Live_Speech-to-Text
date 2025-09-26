import streamlit as st
import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ==========================
# Load ASR Model
# ==========================
@st.cache_resource
def load_asr():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_asr()

# ==========================
# Audio Recorder
# ==========================
def record_audio(duration=5, fs=16000):
    """Record audio from mic"""
    st.info("🎙  Yup!! I'm listerning")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()  # wait until recording finished
    st.success("Hey!!  Recording just finished")
    return np.squeeze(audio)

# ==========================
# Transcription
# ==========================
def transcribe(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]

# ==========================
# Streamlit UI
# ==========================
st.title("Sourav's Live Speech-to-Text (Wav2Vec2)")
st.write("Press the button, speak, and get your words transcribed.")

duration = st.slider("Recording Duration (seconds)", 3, 10, 5)

if st.button("🎙 Click to Record and Transcribe"):
    audio = record_audio(duration=duration)
    st.audio(audio, format="audio/wav", sample_rate=16000)
    text = transcribe(audio)
    st.subheader("Here what you said->:")
    st.write(text)
