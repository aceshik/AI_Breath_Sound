import os
import librosa
import numpy as np
import soundfile as sf

RAW_DIR = "data/raw_audio"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_audio(filename):
    if not filename.endswith(".wav"):
        return
    
    filepath = os.path.join(RAW_DIR, filename)
    y, sr = librosa.load(filepath, sr=16000)
    
    # 무음 제거 + 정규화
    y_trimmed, _ = librosa.effects.trim(y)
    y_normalized = librosa.util.normalize(y_trimmed)

    # 저장
    out_path = os.path.join(PROCESSED_DIR, filename)
    sf.write(out_path, y_normalized, sr)
    print(f"Saved: {out_path}")

def preprocess_all():
    for fname in os.listdir(RAW_DIR):
        preprocess_audio(fname)

if __name__ == "__main__":
    preprocess_all()