import numpy as np
import torch
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# 1. 원본 데이터 로드
X = np.load("data/segments/X.npy")
y = np.load("data/segments/y.npy")  # shape: (n_samples, 2)

# 2. 클래스별 인덱스 분리
idx_normal = np.where((y[:, 0] == 0) & (y[:, 1] == 0))[0]
idx_wheeze = np.where(y[:, 0] == 1)[0]
idx_crackle = np.where(y[:, 1] == 1)[0]

# 3. 클래스 균형 맞추기
undersampled_normal = resample(idx_normal, replace=False, n_samples=2370, random_state=42)
oversampled_crackle = resample(idx_crackle, replace=True, n_samples=2370, random_state=42)

# 4. 새로운 인덱스 조합
final_indices = np.concatenate([undersampled_normal, idx_wheeze, oversampled_crackle])
np.random.shuffle(final_indices)

# 5. 균형 잡힌 데이터셋 생성
X_balanced = X[final_indices]
y_balanced = y[final_indices]

# 6. 정규화 및 텐서 변환
X_balanced = (X_balanced - X_balanced.mean()) / X_balanced.std()
X_balanced = torch.tensor(X_balanced, dtype=torch.float32).unsqueeze(1)
y_balanced = torch.tensor(y_balanced, dtype=torch.float32)

# 7. train/val split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

np.save("data/segments/X_balanced.npy", X_balanced.numpy())
np.save("data/segments/y_balanced.npy", y_balanced.numpy())