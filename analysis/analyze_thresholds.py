import numpy as np
from sklearn.metrics import f1_score, precision_score, classification_report

# 저장된 결과 불러오기
y_true = np.load("results/y_true.npy")
probs = np.load("results/y_pred_probs.npy")

# Wheeze threshold 탐색
print("\n🔬 Wheeze Precision 개선을 위한 Threshold 재탐색:")
for thresh in np.arange(0.25, 0.71, 0.025):
    preds = np.zeros_like(probs)
    preds[:, 0] = probs[:, 0] >= thresh  # Wheeze
    preds[:, 1] = probs[:, 1] >= 0.525   # Crackle 고정
    f1 = f1_score(y_true, preds, average='macro')
    precision = precision_score(y_true[:, 0], preds[:, 0])
    print(f"Wheeze ≥ {thresh:.3f} → Macro F1: {f1:.4f}, Wheeze Precision: {precision:.4f}")

# Crackle threshold 탐색
print("\n🔬 Crackle Precision 개선을 위한 Threshold 재탐색:")
for thresh in np.arange(0.25, 0.71, 0.025):
    preds = np.zeros_like(probs)
    preds[:, 0] = probs[:, 0] >= 0.525   # Wheeze 고정
    preds[:, 1] = probs[:, 1] >= thresh  # Crackle
    f1 = f1_score(y_true, preds, average='macro')
    precision = precision_score(y_true[:, 1], preds[:, 1])
    print(f"Crackle ≥ {thresh:.3f} → Macro F1: {f1:.4f}, Crackle Precision: {precision:.4f}")

# 최종 결과 보고
final_preds = np.zeros_like(probs)
final_preds[:, 0] = probs[:, 0] >= 0.425  # Wheeze
final_preds[:, 1] = probs[:, 1] >= 0.325   # Crackle

print("\n📊 Classification Report (Wheeze, Crackle thresholds = 0.525):")
print(classification_report(y_true, final_preds, target_names=["Wheeze", "Crackle"]))