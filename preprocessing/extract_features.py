import os
import librosa
import numpy as np
import pandas as pd

SEGMENT_DIR = "data/segments"
LABEL_FILE = os.path.join(SEGMENT_DIR, "segments_labels.csv")
N_MFCC = 13
MAX_LEN = 100  # 프레임 수 고정

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    
    return mfcc

def main():
    df = pd.read_csv(LABEL_FILE)
    X = []
    y = []

    for i, row in df.iterrows():
        fname = row["segment"]
        label = [row["wheeze"], row["crackle"]]
        path = os.path.join(SEGMENT_DIR, fname)

        try:
            mfcc = extract_mfcc(path)
            X.append(mfcc)
            y.append(label)
        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(SEGMENT_DIR, "X.npy"), X)
    np.save(os.path.join(SEGMENT_DIR, "y.npy"), y)

    print(f"✅ Feature shape: {X.shape}")
    print(f"✅ Label shape: {y.shape}")
    print("✅ Saved: X.npy, y.npy")

if __name__ == "__main__":
    main()