 # Development of an AI-based respiratory analysis system

## AI 기반 호흡 분석 시스템 개발

### 구조

```bash
AI_Breath_Sound/
├── data/
│   ├── raw_audio/         # 원본 ICBHI 오디오 파일
│   ├── processed/         # 정규화 및 무음 제거된 파일
│   ├── segments/          # 호흡 주기 단위 오디오 조각
│   │   ├── X.npy, y.npy   # MFCC 특징과 라벨 벡터
│   │   └── segments_labels.csv
│   └── icbhi_labels.csv   # 원본 라벨 정보 (start-end 구간)
├── preprocessing/
│   ├── preprocess_icbhi.py    # 오디오 정규화 전처리
│   ├── parse_icbhi_labels.py  # .txt → .csv 라벨 정리
│   ├── split_segments.py      # 호흡 단위로 오디오 분할
│   └── extract_features.py    # MFCC 특징 추출
```

### 현재까지 구현

- ICBHI 2017 데이터셋 구성 분석 및 정리
- '.wav' 오디오 파일 무음 제거 + 정규화
- '.txt' 라벨 파일 파싱 → 'icbhi_labels.csv'
- 호흡 주기 단위로 '.wav' 파일 분할 → 'segments/*.wav'
- 분할된 세그먼트 기반 라벨링 → 'segments_labels.csv'
- 각 세그먼트에서 MFCC 특징 추출 → 'X.npy', 'Y.npy' / 용량이슈로 깃허브 푸시x

