import numpy as np
import os

# 기존 증강용 함수: Gaussian noise 추가
def augment_with_noise(data, num_augments=1, noise_level=0.05):
    augmented = []
    for _ in range(num_augments):
        noise = np.random.normal(0, noise_level, data.shape)
        augmented_sample = data + noise
        augmented.append(augmented_sample)
    return np.array(augmented)

# 데이터 불러오기
X = np.load("data/segments_augmented/X_crackle_augmented.npy")
y = np.load("data/segments_augmented/y_crackle_augmented.npy")

# Wheeze 클래스 선택
wheeze_indices = np.where(y[:, 0] == 1)[0]
X_wheeze = X[wheeze_indices]
y_wheeze = y[wheeze_indices]

print(f"원본 Wheeze 샘플 수: {len(X_wheeze)}")

# 증강 횟수 설정 (예: 1배)
X_aug = augment_with_noise(X_wheeze, num_augments=1)

print("X_aug shape:", X_aug.shape)

X_aug = np.squeeze(X_aug, axis=0)  # (1, 2370, 13, 100) → (2370, 13, 100)

y_aug = np.tile(y_wheeze, (1, 1))  # 동일한 라벨 복제

# 원본 + 증강 데이터 합치기
X_combined = np.concatenate([X, X_aug], axis=0)
y_combined = np.concatenate([y, y_aug], axis=0)

print(f"총 샘플 수 (증강 포함): {len(X_combined)}")

# 저장
os.makedirs("data/segments", exist_ok=True)
np.save("data/segments/X_augmented.npy", X_combined)
np.save("data/segments/y_augmented.npy", y_combined)
print("✅ 증강 데이터 저장 완료: X_augmented.npy, y_augmented.npy")