from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import librosa
import antropy
from scipy.stats import yeojohnson
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import io
from typing import List, Optional
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import IPython.display as ipd
import io
import base64
# Initialize FastAPI app
app = FastAPI(title="Music Emotion Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LyricsEmotionResponse(BaseModel):
    emotion: str
    features: dict
    visualization_base64: str
    waveform_base64: str

class SimpleMusicEmotion:
    def __init__(self):
        self.emotions = ['happy', 'sad', 'energetic', 'calm', 'angry']
    
    def extract_features(self, audio_bytes):
        try:
            # Load audio file
            y, sr = librosa.load(audio_bytes)
            
            # Extract various features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Calculate energy
            energy = float(np.sum(np.abs(y)**2) / len(y))
            
            # Create feature dictionary
            features = {
                'tempo': float(tempo),
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'energy': energy
            }
            
            # Generate visualizations
            # Waveform
            plt.figure(figsize=(12, 4))
            plt.plot(y)
            plt.title('Waveform')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            waveform_buffer = io.BytesIO()
            plt.savefig(waveform_buffer, format='png')
            plt.close()
            waveform_buffer.seek(0)
            waveform_base64 = base64.b64encode(waveform_buffer.getvalue()).decode()
            
            # Mel spectrogram
            mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            mel_buffer = io.BytesIO()
            plt.savefig(mel_buffer, format='png')
            plt.close()
            mel_buffer.seek(0)
            mel_base64 = base64.b64encode(mel_buffer.getvalue()).decode()
            
            return features, waveform_base64, mel_base64
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    def predict_emotion(self, features):
        if features['tempo'] > 150 and features['energy'] > 0.1:
            return 'energetic'
        elif features['tempo'] > 130 and features['spectral_centroid_mean'] > 2000:
            return 'happy'
        elif features['tempo'] < 100 and features['energy'] < 0.05:
            return 'sad'
        elif features['zero_crossing_rate_mean'] > 0.1 and features['energy'] > 0.15:
            return 'angry'
        else:
            return 'calm'



# Response Model
class PredictionResponse(BaseModel):
    predictions: dict
    r2_scores: dict
    emotional_interpretation: str
    visualization_base64: str

def extract_features(audio_bytes):
    """
    Extract audio features from an audio file using librosa.
    
    Args:
        audio_bytes: BytesIO object containing audio data
        
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_bytes)
        
        # Initialize dictionary for features
        features = {}
        
        # Basic features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        features['total_beats'] = len(beats)
        features['average_beats'] = np.mean(beats) if len(beats) > 0 else 0
        
        # Spectral features
        # Chromagram
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_std'] = np.std(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        
        # Chroma CQT
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['chroma_cq_mean'] = np.mean(chroma_cq)
        features['chroma_cq_std'] = np.std(chroma_cq)
        features['chroma_cq_var'] = np.var(chroma_cq)
        
        # Chroma CENS
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        features['chroma_cens_mean'] = np.mean(chroma_cens)
        features['chroma_cens_std'] = np.std(chroma_cens)
        features['chroma_cens_var'] = np.var(chroma_cens)
        
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features['melspectrogram_mean'] = np.mean(mel)
        features['melspectrogram_std'] = np.std(mel)
        features['melspectrogram_var'] = np.var(mel)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features['mfcc_mean'] = np.mean(mfcc)
        features['mfcc_std'] = np.std(mfcc)
        features['mfcc_var'] = np.var(mfcc)
        
        # MFCC Delta
        mfcc_delta = librosa.feature.delta(mfcc)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta_std'] = np.std(mfcc_delta)
        features['mfcc_delta_var'] = np.var(mfcc_delta)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_var'] = np.var(rms)
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['cent_mean'] = np.mean(cent)
        features['cent_std'] = np.std(cent)
        features['cent_var'] = np.var(cent)
        
        # Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spec_bw_mean'] = np.mean(spec_bw)
        features['spec_bw_std'] = np.std(spec_bw)
        features['spec_bw_var'] = np.var(spec_bw)
        
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_std'] = np.std(contrast)
        features['contrast_var'] = np.var(contrast)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)
        features['rolloff_var'] = np.var(rolloff)
        
        # Entropy
        features['entropy_fft'] = antropy.perm_entropy(y)
        features['entropy_welch'] = antropy.spectral_entropy(y, sf=sr)
        
        # Onset Strength
        novelty = librosa.onset.onset_strength(y=y, sr=sr)
        features['novelty_mean'] = np.mean(novelty)
        features['novelty_std'] = np.std(novelty)
        features['novelty_var'] = np.var(novelty)
        
        # Spectral Flatness
        poly = librosa.feature.spectral_flatness(y=y)
        features['poly_mean'] = np.mean(poly)
        features['poly_std'] = np.std(poly)
        features['poly_var'] = np.var(poly)
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        features['tonnetz_var'] = np.var(tonnetz)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_var'] = np.var(zcr)
        
        # Harmonic and Percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        features['harm_mean'] = np.mean(y_harmonic)
        features['harm_std'] = np.std(y_harmonic)
        features['harm_var'] = np.var(y_harmonic)
        
        features['perc_mean'] = np.mean(y_percussive)
        features['perc_std'] = np.std(y_percussive)
        features['perc_var'] = np.var(y_percussive)
        
        # Frame-level statistics
        frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        features['frame_mean'] = np.mean(frames)
        features['frame_std'] = np.std(frames)
        features['frame_var'] = np.var(frames)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure all values are scalar
        for column in feature_df.columns:
            if isinstance(feature_df[column].iloc[0], (list, np.ndarray)):
                feature_df[column] = feature_df[column].apply(lambda x: float(np.mean(x)) if isinstance(x, (list, np.ndarray)) else float(x))
        
        # Handle any remaining NaN values
        feature_df = feature_df.fillna(0)
        
        return feature_df
        
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise

def safe_audio_load(audio_bytes):
    """
    Safely load audio file with error handling.
    
    Args:
        audio_bytes: BytesIO object containing audio data
        
    Returns:
        tuple: (audio_time_series, sampling_rate)
    """
    try:
        y, sr = librosa.load(audio_bytes)
        if len(y) == 0:
            raise ValueError("Empty audio file")
        return y, sr
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        raise


def run_model(feature_set, model='pls', apply_transformation=True, toPredict=['valence', 'energy', 'tension']):
    feature_set.dropna(how='any', axis=0) 
    audio_df = pd.read_csv('features_combined.csv')
    audio_df = audio_df.dropna(how='any', axis=0)
    X = audio_df.loc[:, "tempo":"frame_var"]
    featureName = X.columns.tolist()
    
    if apply_transformation:
        for name in featureName:
            if isinstance(feature_set[name].iloc[0], (list, np.ndarray)):
                feature_set[name] = feature_set[name].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)
            X[name], lam = yeojohnson(X[name])  # Transform the training set
            feature_set[name] = yeojohnson(feature_set[name], lmbda=lam)
    
    # Prepare results
    X = pd.DataFrame(X)
    emotionRatingPrediction = {}
    r2s = {}

    for target in toPredict:
        y = audio_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model == "pls":
            pls = PLSRegression(n_components=10)
            pls.fit(X_train, y_train)
            y_pred = pls.predict(X_test).flatten()
            r2 = pls.score(X_train, y_train)
            pred = pls.predict(feature_set)
        elif model == "lr":
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test).flatten()
            r2 = reg.score(X_train, y_train)
            pred = reg.predict(feature_set)
        elif model == "lr_pca":
            # PCA transformation
            pca = PCA(n_components=11)
            X_train_transformed = pca.fit_transform(X_train)
            X_test_transformed = pca.transform(X_test)
            feature_set_transformed = pca.transform(feature_set)
            
            # Linear regression on transformed data
            reg = LinearRegression()
            reg.fit(X_train_transformed, y_train)
            y_pred = reg.predict(X_test_transformed).flatten()
            r2 = reg.score(X_train_transformed, y_train)
            pred = reg.predict(feature_set_transformed)
        else:
            raise ValueError(f"Unknown model type: {model}")

        # Compute Classification Report
        y_true_binned = np.digitize(y_test, bins=np.linspace(y_test.min(), y_test.max(), 5))
        y_pred_binned = np.digitize(y_pred, bins=np.linspace(y_test.min(), y_test.max(), 5))

        print(f"Classification Report for {target}:")
        print(classification_report(y_true_binned, y_pred_binned))

        emotionRatingPrediction[target] = pred.flatten()
        r2s[target] = r2

    return emotionRatingPrediction, r2s

def generate_emotional_interpretation(predictions: dict) -> str:
    interpretation = "This piece of music appears to be "

    if 'valence' in predictions:
        valence = predictions['valence'][0]
        if valence > 0.7:
            interpretation += "very positive and uplifting"
        elif valence > 0.5:
            interpretation += "somewhat positive"
        elif valence > 0.3:
            interpretation += "somewhat melancholic"
        else:
            interpretation += "rather sad or negative"

    if 'energy' in predictions:
        energy = predictions['energy'][0]
        interpretation += ", with "
        if energy > 0.7:
            interpretation += "high energy and intensity"
        elif energy > 0.5:
            interpretation += "moderate energy levels"
        else:
            interpretation += "calm and relaxed energy"

    if 'tension' in predictions:
        tension = predictions['tension'][0]
        interpretation += ". The music contains "
        if tension > 0.7:
            interpretation += "significant tension and dramatic elements"
        elif tension > 0.5:
            interpretation += "some tension and complexity"
        else:
            interpretation += "minimal tension, flowing smoothly"

    interpretation += "."
    return interpretation

def create_emotion_visualization(predictions: dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    categories = list(predictions.keys())
    values = [predictions[cat][0] for cat in categories]

    values += values[:1]
    categories += categories[:1]

    angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])

    plt.title("Emotional Analysis Results")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return image_base64

@app.get("/")
async def root():
    return {"message": "Music Emotion Analysis API is running"}

@app.post("/analyze", response_model=PredictionResponse)
async def analyze_audio(file: UploadFile = File(...), model_type: str = "pls"):
    try:
        audio_bytes = io.BytesIO(await file.read())
        feature_set = extract_features(audio_bytes)
        
        # Convert any list/array values to scalar values
        for column in feature_set.columns:
            if isinstance(feature_set[column].iloc[0], (list, np.ndarray)):
                feature_set[column] = feature_set[column].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)
        
        if len(feature_set.shape) == 1:
            feature_set = feature_set.to_frame().T
        
        # Ensure no NaN values remain in feature_set
        imputer = SimpleImputer(strategy="mean")
        feature_set = pd.DataFrame(imputer.fit_transform(feature_set), columns=feature_set.columns)
        
        # Ensure all values are scalar
        for col in feature_set.columns:
            if feature_set[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                feature_set[col] = feature_set[col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)

        predictions, r2_scores = run_model(feature_set, model=model_type)
        
        # Convert numpy values to Python native types for JSON serialization
        predictions = {k: v.tolist() if isinstance(v, np.ndarray) else float(v) for k, v in predictions.items()}
        r2_scores = {k: float(v) if isinstance(v, np.ndarray) else v for k, v in r2_scores.items()}
        
        interpretation = generate_emotional_interpretation(predictions)
        visualization = create_emotion_visualization(predictions)


        return PredictionResponse(
            predictions=predictions,
            r2_scores=r2_scores,
            emotional_interpretation=interpretation,
            visualization_base64=visualization
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Import and register routes from the lyrics sentiment module
import lyric_sentiment_analysis as lyrics_sentiment
lyrics_sentiment.register_routes(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
