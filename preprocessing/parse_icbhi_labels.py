import os
import csv

RAW_DIR = "data/raw_audio"
OUTPUT_CSV = "data/icbhi_labels.csv"

def parse_txt_file(txt_path, wav_filename):
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()  # 공백/탭 모두 대응
            if len(parts) != 4:
                print(f"Skipped line in {txt_path}: {line.strip()}")
                continue
            start, end, wheeze, crackle = parts
            rows.append([wav_filename, float(start), float(end), int(wheeze), int(crackle)])
    return rows

def parse_all():
    all_rows = []
    wav_files = set([f.replace(".wav", "") for f in os.listdir(RAW_DIR) if f.endswith(".wav")])

    for file in os.listdir(RAW_DIR):
        if file.endswith(".txt"):
            base = file.replace(".txt", "")
            if base not in wav_files:
                continue  # 설명용 .txt는 무시

            txt_path = os.path.join(RAW_DIR, file)
            wav_filename = base + ".wav"
            rows = parse_txt_file(txt_path, wav_filename)
            all_rows.extend(rows)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "start", "end", "wheeze", "crackle"])
        writer.writerows(all_rows)

    print(f"Saved label CSV: {OUTPUT_CSV} ({len(all_rows)} rows)")

if __name__ == "__main__":
    parse_all()