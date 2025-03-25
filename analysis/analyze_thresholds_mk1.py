import numpy as np
from sklearn.metrics import f1_score, precision_score, classification_report

# 🔁 저장된 결과 불러오기
y_true = np.load("results/y_true.npy")
probs = np.load("results/y_pred_probs.npy")

# 🎯 최적 threshold 탐색
best_f1 = 0.0
best_w, best_c = 0.5, 0.5
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

# 📊 최종 결과
final_preds = np.zeros_like(probs)
final_preds[:, 0] = probs[:, 0] >= best_w
final_preds[:, 1] = probs[:, 1] >= best_c

# 🔍 Crackle 정밀도 집중 탐색
print("\n🔬 Crackle Precision 개선을 위한 Threshold 재탐색:")
for crackle_thresh in np.arange(0.3, 0.61, 0.05):
    temp_preds = final_preds.copy()
    temp_preds[:, 1] = (probs[:, 1] >= crackle_thresh)
    f1 = f1_score(y_true, temp_preds, average='macro')
    precision = precision_score(y_true[:, 1], temp_preds[:, 1])
    print(f"Crackle ≥ {crackle_thresh:.2f} → Macro F1: {f1:.4f}, Crackle Precision: {precision:.4f}")

print("\n📊 Classification Report (best thresholds):")
print(classification_report(y_true, final_preds, target_names=["Wheeze", "Crackle"]))
