import numpy as np
import os
import librosa
import librosa.display
import random

# ✅ 1️⃣ 원본 데이터 불러오기
X = np.load("data/segments_augmented/X_crackle_augmented_advanced.npy")
y = np.load("data/segments_augmented/y_crackle_augmented_advanced.npy")

# ✅ 2️⃣ Wheeze 클래스만 추출
wheeze_indices = np.where(y[:, 0] == 1)[0]  # Wheeze가 있는 샘플만 추출
X_wheeze = X[wheeze_indices]
y_wheeze = y[wheeze_indices]

print(f"원본 Wheeze 샘플 수: {len(X_wheeze)}")

# ✅ 3️⃣ 증강 함수 정의
def add_noise(x, noise_level=0.01):
    """ 가우시안 노이즈 추가 """
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

def time_mask(x, max_width=10):
    """ Time Masking (일부 시간 구간 0으로 설정) """
    x = x.copy()
    t = x.shape[-1]
    width = np.random.randint(1, max_width)
    start = np.random.randint(0, t - width)
    x[:, start:start + width] = 0
    return x

def pitch_shift(x, sample_rate=22050, n_steps=1):
    """ 피치 변형 (높이 변경, Wheeze는 변화량을 줄임) """
    return librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=n_steps)

def add_reverb(x, sample_rate=22050):
    """ 잔향 효과 추가 (Reverberation) """
    return librosa.effects.preemphasis(x, coef=0.95)

def mixup(x1, x2, alpha=0.2):
    """ Mixup (두 샘플을 혼합) """
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2

# ✅ 4️⃣ Wheeze 증강 수행
augmented = []
for x in X_wheeze:
    # 기존 방식 유지
    augmented.extend([
        add_noise(x),
        time_mask(x),
    ])
    # 새로운 방식 추가
    augmented.extend([
        pitch_shift(x),
        add_reverb(x),
    ])

# ✅ 5️⃣ Mixup 적용 (Wheeze 샘플 2개를 랜덤으로 섞음)
mixup_aug = []
for _ in range(len(X_wheeze)):
    i, j = np.random.choice(len(X_wheeze), 2, replace=False)
    mixup_aug.append(mixup(X_wheeze[i], X_wheeze[j]))

# ✅ 6️⃣ 최종 데이터 구성
X_aug = np.stack(augmented + mixup_aug)
y_aug = np.tile(y_wheeze, (len(X_aug) // len(y_wheeze), 1))  # 증강된 샘플 수에 맞게 레이블 복제

# ✅ 7️⃣ 원본 + 증강 데이터 합치기 및 저장
X_new = np.concatenate([X, X_aug], axis=0)
y_new = np.concatenate([y, y_aug], axis=0)

os.makedirs("data/segments_augmented", exist_ok=True)
np.save("data/segments_augmented/X_all_augmented_advanced.npy", X_new)
np.save("data/segments_augmented/y_all_augmented_advanced.npy", y_new)

print("✅ Wheeze 데이터 증강 완료. 저장 경로: data/segments_augmented/")
print("✅ 최종 X shape:", X_new.shape)
print("✅ 최종 y shape:", y_new.shape)