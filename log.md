# AI 기반 호흡음 분석 모델 실험 기록

## ✅ 모델 구성

- **기본 모델**: CNN 기반 이진 다중 레이블 분류 (Wheeze / Crackle)
- **확장 모델**: CNN + LSTM 구조로 시간적 특성 반영
- **입력 데이터**: MFCC 기반 13×100 Segment
- **데이터 전처리**:
  - Z-score 정규화
  - 불균형 조정 (Undersampling + Oversampling)

---

## 🔁 실험 기록 요약

### 실험 #1: CNN 기본 구조
- **Dropout 제거**
- Wheeze: Precision 0.44, Recall 0.85  
- Crackle: Precision 0.27, Recall 0.51  
- **Macro F1**: 0.46

### 실험 #3: 데이터 불균형 조정
- Wheeze: Precision 0.54, Recall 0.90  
- Crackle: Precision 0.52, Recall 0.76  
- **Macro F1**: 0.64

### 실험 #4-1: CNN 구조 심화 + Dropout 추가
- Macro F1: 0.64 유지

### 실험 #4-2: 동일 구조 + Dropout 제거
- Wheeze: Precision 0.59, Recall 0.79  
- Crackle: Precision 0.47, Recall 0.90  
- **Macro F1**: 0.64

---

## 🧠 LSTM 도입 이후 (CNN + LSTM)

### 실험 #5: CNN + LSTM + 불균형 조정 + Threshold 튜닝
- Wheeze: Precision 0.54, Recall 0.89  
- Crackle: Precision 0.50, Recall 0.86  
- **Macro F1**: 0.65  
- 최적 Thresholds: Wheeze 0.25, Crackle 0.25

### Confusion Matrix
|**Wheeze**     | Predicted Negative | Predicted Positive |
|---------------|--------------------|---------------------|
| Actual Negative | 154              | 633                 |
| Actual Positive | 29               | 606                 |

|**Crackle**    | Predicted Negative | Predicted Positive |
|---------------|--------------------|---------------------|
| Actual Negative | 248              | 600                 |
| Actual Positive | 72               | 502                 |

→ Recall은 높으나 Precision은 낮음 → **FP 많음**

---

## 📈 Crackle Precision 개선 실험

| Crackle Threshold | Macro F1 | Crackle Precision |
|-------------------|----------|--------------------|
| 0.250             | 0.6468   | 0.4995             |
| 0.275             | 0.6447   | 0.5058             |
| 0.300             | 0.6427   | 0.5135             |
| 0.325             | 0.6378   | 0.5186             |
| 0.350             | 0.6380   | 0.5267             |
| 0.375             | 0.6359   | 0.5346             |
| 0.400             | 0.6322   | 0.5416             |
| 0.425             | 0.6340   | 0.5654             |
| 0.450             | 0.6292   | 0.5794             |
| 0.475             | 0.6262   | 0.6033             |
| 0.500             | 0.6182   | 0.6134             |
| 0.525             | 0.6159   | 0.6393             |
| 0.550             | 0.6008   | 0.6488             |
| 0.575             | 0.5924   | 0.6649             |
| 0.600             | 0.5755   | 0.6790             |
| 0.625             | 0.5592   | 0.6926             |
| 0.650             | 0.5486   | 0.7070             |
| 0.675             | 0.5262   | 0.6996             |
| 0.700             | 0.5148   | 0.7079             |

---

## 🗺️ 현재 진행 상황 요약

- [x] CNN 구조 실험 완료
- [x] 데이터 불균형 조정 완료
- [x] CNN + LSTM 구조 구현 및 학습 완료
- [x] 최적 threshold 튜닝 완료
- [x] Crackle precision 개선을 위한 threshold 미세 조정 진행
- [x] Confusion matrix 분석
- [ ] FP 샘플 시각화
- [ ] 모델 개선 방향 도출 (Feature 또는 구조 개선)

