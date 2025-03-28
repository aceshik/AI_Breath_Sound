import numpy as np
import os

# 🔧 증강 함수 (Noise + Time Mask만 사용)
def add_noise(x, noise_level=0.02):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

def time_mask(x, max_width=10):
    x = x.copy()
    t = x.shape[-1]
    width = np.random.randint(1, max_width)
    start = np.random.randint(0, t - width)
    x[:, start:start + width] = 0
    return x

# 데이터 불러오기
X = np.load("data/segments/X.npy")
y = np.load("data/segments/y.npy")

def augment_class(class_index, name):
    indices = np.where(y[:, class_index] == 1)[0]
    X_class = X[indices]
    y_class = y[indices]

    augmented = []
    for x in X_class:
        augmented.append(add_noise(x))
        augmented.append(time_mask(x))

    X_aug = np.stack(augmented)
    y_aug = np.tile(y_class, (2, 1))  # 2배 증강

    X_new = np.concatenate([X, X_aug], axis=0)
    y_new = np.concatenate([y, y_aug], axis=0)

    os.makedirs("data/segments_augmented_refined", exist_ok=True)
    np.save(f"data/segments_augmented_refined/X_{name}_augmented.npy", X_new)
    np.save(f"data/segments_augmented_refined/y_{name}_augmented.npy", y_new)

    print(f"✅ {name} 증강 완료. 최종 샘플 수: {len(X_new)}")

augment_class(0, "wheeze")
augment_class(1, "crackle")