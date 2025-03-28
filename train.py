import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import copy
from models.cnn_lstm_model import CNNLSTM, save_model  # 모델 및 저장 함수 불러오기

# 데이터 불러오기 및 전처리
X = np.load("data/segments_augmented_refined/X_all_augmented_refined.npy")
y = np.load("data/segments_augmented_refined/y_all_augmented_refined.npy")

X = (X - X.mean()) / X.std()  # 정규화
X = torch.tensor(X, dtype=torch.float32)

# 차원 자동 보정: Conv2D는 (N, C, H, W) → C=1이어야 하므로
if X.ndim == 3:  
    X = X.unsqueeze(1)
elif X.ndim == 5:  
    X = X.squeeze(1).squeeze(1)

y = torch.tensor(y, dtype=torch.float32)

# Train/Validation 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# DataLoader 구성
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)


# 모델 선택

from models.cnn_lstm_model import CNNLSTM

model_name = "cnn_lstm"  # 실험할 모델을 선택

if model_name == "cnn_lstm":
    model = CNNLSTM()
else:
    raise ValueError("올바른 모델을 선택하세요!")



# 손실 함수 및 옵티마이저 설정
pos_weight = torch.tensor([1.0, 1.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 학습 루프
EPOCHS = 20
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_model_state = None
patience = 5
wait = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # Early Stopping
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

# Best 모델 저장
model.load_state_dict(best_model_state)
save_model(model, "models/cnn_lstm_model.pth")  # 모델 저장

# 모델 평가 및 학습 결과 저장
model.eval()
all_probs, all_labels = [], []

with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        all_probs.append(torch.sigmoid(pred))
        all_labels.append(yb)

# numpy 변환
probs = torch.cat(all_probs).numpy()
y_true = torch.cat(all_labels).numpy().astype(int)

# ✅ 학습 결과 저장
np.save(f"results/{model_name}_y_true.npy", y_true)
np.save(f"results/{model_name}_y_pred_probs.npy", probs)
np.save(f"results/{model_name}_train_losses.npy", np.array(train_losses))
np.save(f"results/{model_name}_val_losses.npy", np.array(val_losses))

print(f"✅ 학습 결과 저장 완료: results/{model_name}_*.npy")