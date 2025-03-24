import os
import librosa
import soundfile as sf
import pandas as pd

RAW_DIR = "data/raw_audio"
SEGMENT_DIR = "data/segments"
LABEL_CSV = "data/icbhi_labels.csv"
SAMPLE_RATE = 16000

os.makedirs(SEGMENT_DIR, exist_ok=True)

def split_segment(row, count):
    filename = row["filename"]
    start = row["start"]
    end = row["end"]
    wheeze = row["wheeze"]
    crackle = row["crackle"]

    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {filename}")
        return None

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]

        segment_filename = f"{filename.replace('.wav', '')}_{count}.wav"
        segment_path = os.path.join(SEGMENT_DIR, segment_filename)

        sf.write(segment_path, segment, sr)
        return {
            "segment": segment_filename,
            "wheeze": wheeze,
            "crackle": crackle
        }

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return None

def main():
    df = pd.read_csv(LABEL_CSV)
    segment_rows = []

    count_map = {}

    for _, row in df.iterrows():
        fname = row["filename"]
        count_map[fname] = count_map.get(fname, 0) + 1
        seg_info = split_segment(row, count_map[fname])
        if seg_info:
            segment_rows.append(seg_info)

    # Save labels for segments
    out_df = pd.DataFrame(segment_rows)
    out_df.to_csv(os.path.join(SEGMENT_DIR, "segments_labels.csv"), index=False)
    print(f"Segments saved to {SEGMENT_DIR}")
    print(f"Label file saved to segments_labels.csv ({len(out_df)} segments)")

if __name__ == "__main__":
    main()