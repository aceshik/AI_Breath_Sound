import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 및 전처리 (불균형 조정된 버전 사용 + crackle 증강)
X = np.load("data/segments_augmented/X_balanced.npy")
y = np.load("data/segments_augmented/y_balanced.npy")

X = (X - X.mean()) / X.std()  # 정규화
X = torch.tensor(X, dtype=torch.float32)

# 차원 자동 보정: Conv2D는 (N, C, H, W) → C=1이어야 하므로
if X.ndim == 3:         # (N, 13, 100) → unsqueeze
    X = X.unsqueeze(1)
elif X.ndim == 5:       # (N, 1, 1, 13, 100) → squeeze 두 번
    X = X.squeeze(1).squeeze(1)

print("✅ 최종 X shape:", X.shape)  # 디버깅용

y = torch.tensor(y, dtype=torch.float32)

# Train/Val 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# DataLoader 구성
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# 2. CNN + LSTM 모델 정의
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 2)  # bidirectional
        )

    def forward(self, x):
        x = self.cnn(x)  # (B, 64, H, W) → 예: (B, 64, 5, 5)
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, seq_len, input_size) = (B, 25, 64)

        x, _ = self.lstm(x)  # input_size=64이므로 맞음
        x = self.fc(x[:, -1])  # 마지막 timestep만 사용
        return x

model = CNNLSTM()
pos_weight = torch.tensor([1.0, 1.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 학습 루프
EPOCHS = 20
train_losses, val_losses = [], []

import copy

# EarlyStopping 설정
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
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()
            all_probs.append(torch.sigmoid(pred))
            all_labels.append(yb)
    val_losses.append(val_loss / len(val_loader))
    
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # EarlyStopping 체크
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

# Best 모델 로드
model.load_state_dict(best_model_state)

# 4. 평가 및 저장
probs = torch.cat(all_probs).numpy()
y_true = torch.cat(all_labels).numpy().astype(int)

# 🔒 학습 결과 저장 (추후 재사용)
np.save("results/y_true.npy", y_true)
np.save("results/y_pred_probs.npy", probs)
np.save("results/train_losses.npy", np.array(train_losses))
np.save("results/val_losses.npy", np.array(val_losses))
print("✅ 학습 결과 저장 완료.")

'''

# Threshold 튜닝
best_w, best_c, best_f1 = 0.5, 0.5, 0.0
for w in np.arange(0.2, 0.61, 0.05):
    for c in np.arange(0.2, 0.61, 0.05):
        preds = np.zeros_like(probs)
        preds[:, 0] = probs[:, 0] >= w
        preds[:, 1] = probs[:, 1] >= c
        f1 = f1_score(y_true, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_w, best_c = w, c

print(f"\n✅ 최적 Thresholds → Wheeze: {best_w:.2f}, Crackle: {best_c:.2f} (Macro F1: {best_f1:.4f})")

# 복사 전에 NumPy 배열로 변환 (안 되어 있다면)
if isinstance(final_preds, torch.Tensor):
    final_preds = final_preds.numpy()

print("\n🔬 Crackle Precision 개선을 위한 Threshold 재탐색:")
for crackle_thresh in np.arange(0.3, 0.61, 0.05):
    temp_preds = final_preds.copy()
    temp_preds[:, 1] = (probs[:, 1] >= crackle_thresh)
    f1 = f1_score(y_true, temp_preds, average='macro')
    precision = precision_score(y_true[:, 1], temp_preds[:, 1])
    print(f"Crackle ≥ {crackle_thresh:.2f} → Macro F1: {f1:.4f}, Crackle Precision: {precision:.4f}")

final_preds = np.zeros_like(probs)
final_preds[:, 0] = probs[:, 0] >= best_w
final_preds[:, 1] = probs[:, 1] >= best_c

print("\n📊 Classification Report (best thresholds):")
print(classification_report(y_true, final_preds, target_names=["Wheeze", "Crackle"]))

np.save("results/y_true.npy", y_true)
np.save("results/y_pred.npy", final_preds)
np.save('results/y_pred_probs.npy', probs)

# 손실 곡선 시각화
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title("Loss Curve (CNN + LSTM)")
plt.show()
'''