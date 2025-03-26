import numpy as np
import torch
from sklearn.metrics import classification_report
from models.cnn_lstm_model import load_model

# threshold 설정
threshold = 0.7

# 평가할 모델 선택
model_name = "cnn_lstm" 
model_name_model = "cnn_lstm_model"

# 평가할 데이터 선택 (True: 증강 데이터, False: 원본 데이터)
use_augmented_data = False 

# 저장된 모델 불러오기
model = load_model(model_name, f"models/{model_name_model}.pth")

# 데이터 로드
if use_augmented_data:
    print("증강된 데이터로 평가를 진행합니다.")
    X = np.load("data/segments_augmented/X_augmented.npy")
    y = np.load("data/segments_augmented/y_augmented.npy")
else:
    print("원본 데이터로 평가를 진행합니다.")
    X = np.load("data/segments/X.npy")
    y = np.load("data/segments/y.npy")


# 데이터 전처리 (정규화 및 PyTorch Tensor 변환)
X = (X - X.mean()) / X.std()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 차원 자동 보정: Conv2D 입력 맞추기
if X.ndim == 3:
    X = X.unsqueeze(1)
elif X.ndim == 5:
    X = X.squeeze(1).squeeze(1)

# Train/Test 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 모델 예측 수행
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred = torch.sigmoid(y_test_pred).numpy()  # 확률값 변환
    y_test_pred = (y_test_pred > 0.5).astype(int)  # 임계값 0.5 기준 이진 분류

# 성능 평가
print("\n=== Test Set Performance ===")
print(classification_report(y_test.numpy().astype(int), y_test_pred, digits=4))

# 평가 결과 저장
np.save(f"results/{model_name}_y_true.npy", y_test.numpy().astype(int))
np.save(f"results/{model_name}_y_pred.npy", y_test_pred)
print(f"✅ 평가 결과 저장 완료: results/{model_name}_y_true.npy & results/{model_name}_y_pred.npy")