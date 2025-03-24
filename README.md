 # AI 기반 호흡 분석 시스템 개발
## Development of an AI-based respiratory analysis system
### 김문종, 원영식, 최정연

### 구조

```bash
AI_BREATH_SOUND/
├── .venv/                        # 가상환경 (Python 3.13)
│
├── analysis/                     # 분석 스크립트
│   ├── analyze_confusion.py
│   ├── analyze_thresholds.py
│   └── analyze_thresholds_mk1.py
│
├── augmentation/                # 데이터 증강
│   ├── augment_crackle.py
│   └── augment_wheeze.py
│
├── data/                         # 데이터셋 및 전처리 결과 저장
│   ├── processed/                # 전처리된 중간 데이터(ex.정규화, 필터링된 데이터)
│   ├── raw_audio/                # 원본 오디오 데이터(ICBHI 2017 등)
│   ├── segments/                 # 원본 세그먼트 데이터(X.npy, y.npy etc)
│   └── segments_augmented/       # 증강된 segment 데이터들
│       ├── X_augmented.npy
│       ├── X_balanced.npy
│       ├── y_augmented.npy
│       └── y_balanced.npy
│
├── images/                       # 시각화 결과 이미지 (confusion matrix 등)
│
├── models/                       # 모델 구조 및 학습 스크립트(굳이 필요 없는 파일들 날림)
│   ├── cnn_lstm_model.py
│   ├── cnn_lstm_model_mk1.py
│   ├── cnn_lstm_train.py
│   ├── cnn_train.py
│   ├── cnn_train_advanced.py
│   ├── evaluate_mlp.py
│   ├── mlp_train.py
│   └── train_model.py            
│
├── notebooks/                    # (비어있음 또는 Jupyter 작업 공간)
│
├── preprocessing/                # 데이터 전처리 파일들
│   ├── balance_sampling.py
│   ├── extract_features.py
│   ├── parse_icbhi_labels.py
│   ├── preprocess_icbhi.py
│   └── split_segments.py
│
├── results/                      # 예측 결과, 학습결과
│   ├── train_losses.npy
│   ├── val_losses.npy
│   ├── y_pred_probs.npy
│   └── y_true.npy
│
├── .gitignore
├── log.md                        # 실험 로그 메모
├── notepad.py                    # 임시 스크립트 / 실험용
├── README.md                     # 프로젝트 설명
└── requirements.txt              # 패키지 의존성 목록
```

### 현재까지 구현

- ICBHI 2017 데이터셋 구성 분석 및 정리
- '.wav' 오디오 파일 무음 제거 + 정규화
- '.txt' 라벨 파일 파싱 → 'icbhi_labels.csv'
- 호흡 주기 단위로 '.wav' 파일 분할 → 'segments/*.wav'
- 분할된 세그먼트 기반 라벨링 → 'segments_labels.csv'
- 각 세그먼트에서 MFCC 특징 추출 → 'X.npy', 'Y.npy' / 용량이슈로 깃허브 푸시x




## MLP_TRAIN

초기 MLP 모델에서 학습이 제대로 진행되지 않음 → 하이퍼파라미터, 전처리 조정

### 초기모델(학습률: 0.001, 정규화 없음)
- 학습률(lr): 0.001
- 입력 정규화 없음
- Loss가 점점 커지며 27.3까지 증가
- 모델 발산하여 학습 실패

<details>
<summary> 초기 MLP LOSS양 </summary>
[1/20] Train Loss: 1.8743 | Val Loss: 21.0850<br>
[2/20] Train Loss: 21.2713 | Val Loss: 24.0106<br>
[3/20] Train Loss: 24.2620 | Val Loss: 24.3964<br>
[4/20] Train Loss: 24.6545 | Val Loss: 23.5934<br>
[5/20] Train Loss: 23.8299 | Val Loss: 20.7134<br>
[6/20] Train Loss: 20.8538 | Val Loss: 34.3102<br>
[7/20] Train Loss: 33.8169 | Val Loss: 22.5947<br>
[8/20] Train Loss: 22.7939 | Val Loss: 25.2366<br>
[9/20] Train Loss: 25.5058 | Val Loss: 26.2734<br>
[10/20] Train Loss: 26.5680 | Val Loss: 26.6861<br>
[11/20] Train Loss: 27.0117 | Val Loss: 26.8341<br>
[12/20] Train Loss: 27.1842 | Val Loss: 26.8880<br>
[13/20] Train Loss: 27.2585 | Val Loss: 26.9114<br>
[14/20] Train Loss: 27.2955 | Val Loss: 26.9172<br>
[15/20] Train Loss: 27.3090 | Val Loss: 26.9243<br>
[16/20] Train Loss: 27.3177 | Val Loss: 26.9269<br>
[17/20] Train Loss: 27.3239 | Val Loss: 26.9292<br>
[18/20] Train Loss: 27.3270 | Val Loss: 26.9313<br>
[19/20] Train Loss: 27.3287 | Val Loss: 26.9332<br>
[20/20] Train Loss: 27.3309 | Val Loss: 26.9350<br>
</details>

### 개선된 모델(학습률 감소 + 정규화)
- 학습률(lr): 0.0001 (1e-4)
- 입력 정규화 추가:
```python
  X = (X - X.mean()) / X.std()
  ```

<details>
<summary> 개선 MLP LOSS양 </summary>
[1/20] Train Loss: 0.6941 | Val Loss: 0.6722<br>
[2/20] Train Loss: 0.6721 | Val Loss: 0.6537<br>
[3/20] Train Loss: 0.6542 | Val Loss: 0.6381<br>
[4/20] Train Loss: 0.6390 | Val Loss: 0.6249<br>
[5/20] Train Loss: 0.6263 | Val Loss: 0.6137<br>
[6/20] Train Loss: 0.6155 | Val Loss: 0.6042<br>
[7/20] Train Loss: 0.6064 | Val Loss: 0.5963<br>
[8/20] Train Loss: 0.5991 | Val Loss: 0.5903<br>
[9/20] Train Loss: 0.5937 | Val Loss: 0.5861<br>
[10/20] Train Loss: 0.5901 | Val Loss: 0.5835<br>
[11/20] Train Loss: 0.5882 | Val Loss: 0.5821<br>
[12/20] Train Loss: 0.5874 | Val Loss: 0.5815<br>
[13/20] Train Loss: 0.5872 | Val Loss: 0.5810<br>
[14/20] Train Loss: 0.5872 | Val Loss: 0.5805<br>
[15/20] Train Loss: 0.5870 | Val Loss: 0.5798<br>
[16/20] Train Loss: 0.5864 | Val Loss: 0.5788<br>
[17/20] Train Loss: 0.5855 | Val Loss: 0.5776<br>
[18/20] Train Loss: 0.5843 | Val Loss: 0.5762<br>
[19/20] Train Loss: 0.5828 | Val Loss: 0.5747<br>
[20/20] Train Loss: 0.5812 | Val Loss: 0.5732<br>
</details>

![개선 MLP LOSS 그래프](images/MLP_2_loss.png)

### 개선된 모델 평가, 데이터 불균형으로 모두 정상으로 예측
### 목표: 예측 성능을 최대한 향상시키는 것

###  실험 #1: Dropout 제거
- 구조: 기본 CNN
- 결과:
  - Wheeze Precision: 0.44 → Recall: 0.85
  - Crackle Precision: 0.27 → Recall: 0.51
  - **Macro F1-score: 0.46 (소폭 향상)**

---

###  실험 #2: 데이터 불균형 완화
- 방법: Undersampling + Oversampling 조합
- 결과:
  - Wheeze Precision: 0.54 / Recall: 0.90
  - Crackle Precision: 0.52 / Recall: 0.76
  - **Macro F1-score: 0.64 (의미 있는 향상)**

---

###  실험 #3: CNN 구조 조정
- 구조: 필터 수 증가 + Dropout 추가/제거 비교

#### 3-1. Dropout 포함
- Macro F1-score: 0.64

#### 3-2. Dropout 제거
- Wheeze Precision: 0.59 / Recall: 0.79  
- Crackle Precision: 0.47 / Recall: 0.90  
- **Macro F1-score: 0.64 (동일)**  
- **Dropout 제거 시 Precision이 더 높음** → 선택

### LSTM 추가 (CNN + LSTM)
- CNN만으로는 성능 향상 한계 도달
- 데이터: 불균형 조정된 버전 사용
- 학습시간: 복잡성이 커져 더 오래걸림
- 초기 손실량: 0.6828(확연히 낮아짐)

---

### 새로운 데이터 전처리
데이터 불균형 조정
- X_balanced.npy, y_balanced.npy 사용 (undersampling + oversampling)
- 정규화 (X - mean) / std
- 차원 보정: (N, 13, 100) → (N, 1, 13, 100)
- Train/Val 분리: test_size = 0.2, random_state = 42

### 모델 구조: CNN + BiLSTM

```bash
Conv2d → ReLU → MaxPool2d  
Conv2d → ReLU → MaxPool2d  
↓  
Reshape to (batch, seq_len, input_size)  
↓  
Bidirectional LSTM  
↓  
Linear → 2 logits (Wheeze, Crackle)
```
- LSTM input size: 64, hidden size: 64, bidirectional
- BCEWithLogitsLoss 사용 (pos_weight 조정 가능)

### 학습 설정
- Optimizer: Adam(lr=)
- Epochs: 20
- Batch size: 32
- 손실 기록 및 시각화 저장

### Threshold 튜닝
- 탐색 범위: [0.2, 0.25, ..., 0.60]
- 기준: Macro F1 최재화
- 최적 결과
```bash
Wheeze ≥ 0.25, Crackle ≥ 0.25 → Macro F1 ≈ 0.6229
```
Crackle Precision이 계속 너무 낮게 나옴 →

### Crackle Precision 개선
- analysis/analyze_crackle_errors.py
- False Positive 예시 10건 출력
- 별도 분석 코드: analyze_thresholds.py 생성

### Crackle Threshold 재탐색 결과

| Crackle Threshold | Macro F1 | Crackle Precision |
|-------------------|----------|-------------------|
| 0.30              | 0.6123   | 0.4072            |
| 0.35              | 0.5468   | 0.5000            |
| 0.40              | 0.5072   | 0.5632            |
| 0.45              | 0.4692   | 0.6193            |
| 0.50              | 0.4306   | 0.6218            |
| 0.55              | 0.4075   | 0.6548            |
| 0.60              | 0.3855   | 0.6610            |



#### 일단 여기까지 했음


models/cnn_lstm_model.py 에서 모델 학습 한 파일 results/ 에 저장하고
그거 불러와서
analysis/analyze_thresholds.py 에서 thresholds 뭘로 하는게 제일 나은지 계속 돌려야함

crackle 증강하니까 엄청 잘나오고
wheeze는 원래 약했어서
wheeze까지 증강했는데

wheeze 개쎄지고
crackle 개약해짐

개수가 안맞는거같아서 crackle 개수 늘리는 방향으로 진행해야할듯

아 그리고 증강된 .npy augmented 에 안들어가있고 그냥 segments에 들어가있을거같은데 그거 업데이트하기