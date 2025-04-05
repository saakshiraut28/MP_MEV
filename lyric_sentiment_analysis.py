import io
import os
import uuid
import base64
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import noisereduce as nr
import torch
import whisper
from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from transformers import pipeline
from pydub import AudioSegment
from pydub.utils import which
import textwrap

# Set up FFmpeg path for pydub
ffmpeg_path = which("ffmpeg")
if ffmpeg_path is None:
    raise EnvironmentError("FFmpeg not found. Please install it and add to PATH.")
AudioSegment.converter = ffmpeg_path

# Create temp directory
os.makedirs("temp", exist_ok=True)

# Response model
class LyricsSentimentResponse(BaseModel):
    lyrics: str
    basic_sentiment: str
    roberta_sentiment: str
    emotions: Dict[str, float]
    visualization_base64: Optional[str] = None
    waveform_base64: Optional[str] = None

# Helper: Convert MP3 to WAV
def convert_mp3_to_wav(mp3_bytes, wav_path):
    temp_mp3_path = wav_path.replace(".wav", ".mp3")
    with open(temp_mp3_path, 'wb') as f:
        f.write(mp3_bytes)
    audio = AudioSegment.from_file(temp_mp3_path, format="mp3")
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    os.remove(temp_mp3_path)

# Noise reduction
def remove_background_noise(wav_path, output_path, fast=True):
    y, sr = librosa.load(wav_path, sr=16000)
    if fast:
        sf.write(output_path, y, sr)
    else:
        reduced_noise = nr.reduce_noise(y=y, sr=sr)
        sf.write(output_path, reduced_noise, sr)

# Transcribe
def transcribe_with_whisper(wav_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base").to(device)
    result = model.transcribe(wav_path)
    return result["text"]

# Basic sentiment
def analyze_basic_sentiment(lyrics):
    pipeline_basic = pipeline("sentiment-analysis")
    chunks = textwrap.wrap(lyrics, 500)
    results = [pipeline_basic(chunk)[0] for chunk in chunks]
    pos = sum(1 for r in results if r["label"] == "POSITIVE")
    neg = sum(1 for r in results if r["label"] == "NEGATIVE")
    return "POSITIVE" if pos > neg else "NEGATIVE"

# RoBERTa sentiment
def analyze_roberta_sentiment(lyrics):
    pipeline_roberta = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    chunks = textwrap.wrap(lyrics, 500)
    results = [pipeline_roberta(chunk)[0] for chunk in chunks]
    pos = sum(1 for r in results if r["label"] == "positive")
    neg = sum(1 for r in results if r["label"] == "negative")
    neu = sum(1 for r in results if r["label"] == "neutral")
    if pos > max(neg, neu): return "POSITIVE"
    if neg > max(pos, neu): return "NEGATIVE"
    return "NEUTRAL"

# Emotion analysis
def analyze_emotions(lyrics):
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    chunks = textwrap.wrap(lyrics, 500)
    results = [emotion_pipeline(chunk)[0] for chunk in chunks]
    emotions = {}
    for r in results:
        label = r["label"]
        score = r["score"]
        emotions[label] = (emotions[label] + score) / 2 if label in emotions else score
    return emotions

# Visualization
def generate_visualizations(audio_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        # Waveform
        plt.figure(figsize=(12, 3))
        plt.plot(y)
        plt.title('Waveform')
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        plt.close()
        buf1.seek(0)
        waveform_b64 = base64.b64encode(buf1.read()).decode()

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(12, 3))
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        plt.close()
        buf2.seek(0)
        mel_b64 = base64.b64encode(buf2.read()).decode()
        return waveform_b64, mel_b64
    except Exception as e:
        print(f"Visualization error: {e}")
        return "", ""

# Core handler
async def handle_lyrics_analysis(file: UploadFile, fast_mode: bool = True):
    try:
        audio_bytes = await file.read()
        temp_id = str(uuid.uuid4())
        wav_path = f"temp/{temp_id}.wav"
        cleaned_wav = f"temp/{temp_id}_clean.wav"
        convert_mp3_to_wav(audio_bytes, wav_path)
        remove_background_noise(wav_path, cleaned_wav, fast=fast_mode)
        lyrics = transcribe_with_whisper(cleaned_wav)
        basic = analyze_basic_sentiment(lyrics)
        roberta = analyze_roberta_sentiment(lyrics)
        emotions = analyze_emotions(lyrics)
        waveform_b64, mel_b64 = generate_visualizations(audio_bytes)
        os.remove(wav_path)
        os.remove(cleaned_wav)
        return LyricsSentimentResponse(
            lyrics=lyrics,
            basic_sentiment=basic,
            roberta_sentiment=roberta,
            emotions=emotions,
            waveform_base64=waveform_b64,
            visualization_base64=mel_b64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register FastAPI routes
def register_routes(app):
    @app.post("/analyze-lyrics-sentiment", response_model=LyricsSentimentResponse)
    async def analyze_lyrics(file: UploadFile = File(...), fast_mode: bool = True):
        return await handle_lyrics_analysis(file, fast_mode)
