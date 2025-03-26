import numpy as np
import os

# 1. 원본 데이터 불러오기
X = np.load("data/segments/X_balanced.npy")
y = np.load("data/segments/y_balanced.npy")

# 2. Crackle 클래스만 추출
crackle_indices = np.where(y[:, 1] == 1)[0]
X_crackle = X[crackle_indices]
y_crackle = y[crackle_indices]

# 3. 증강 함수 정의
def add_noise(x, noise_level=0.01):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

def time_mask(x, max_width=10):
    x = x.copy()
    t = x.shape[-1]
    width = np.random.randint(1, max_width)
    start = np.random.randint(0, t - width)
    x[:, start:start + width] = 0
    return x

def freq_mask(x, max_width=3):
    x = x.copy()
    f = x.shape[-2]
    width = np.random.randint(1, max_width)
    start = np.random.randint(0, f - width)
    x[start:start + width, :] = 0
    return x

# 4. 증강 수행
augmented = []
for x in X_crackle:
    augmented.extend([
        add_noise(x),
        time_mask(x),
        freq_mask(x)
    ])

X_aug = np.stack(augmented)
y_aug = np.tile(y_crackle, (3, 1))

# 5. 합치기 및 저장
X_new = np.concatenate([X, X_aug], axis=0)
y_new = np.concatenate([y, y_aug], axis=0)

os.makedirs("data/segments_augmented", exist_ok=True)
np.save("data/segments_augmented/X_crackle_augmented.npy", X_new)
np.save("data/segments_augmented/y_crackle_augmented.npy", y_new)

print("증강 완료. 저장 경로: data/segments_augmented/")
print("최종 X shape:", X_new.shape)
print("최종 y shape:", y_new.shape)