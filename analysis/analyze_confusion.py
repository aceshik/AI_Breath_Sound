import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 파일 불러오기
y_true = np.load("results/y_true.npy")
y_pred = np.load("results/y_pred.npy")

# 클래스 이름
labels = ["Wheeze", "Crackle"]

# 각 클래스에 대해 confusion matrix 계산
for i, label in enumerate(labels):
    cm = confusion_matrix(y_true[:, i], y_pred[:, i])
    print(f"Confusion Matrix for {label}:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {label}")
    plt.tight_layout()
    plt.show()