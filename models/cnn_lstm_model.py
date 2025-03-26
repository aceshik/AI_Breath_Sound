import torch
import torch.nn as nn
import os

# CNN + LSTM 모델
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
        self.fc = nn.Linear(64 * 2, 2)  # bidirectional

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

# 모델 저장 함수
def save_model(model, path="models/cnn_lstm_model.pth"):
    """ PyTorch 모델 저장 (폴더 자동 생성 + 디버깅 메시지 추가) """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 폴더 없으면 생성
        torch.save(model.state_dict(), path)
        print(f"✅ 모델이 {path} 에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")

# 모델 불러오기 함수
def load_model(model_name="cnn_lstm", path="models/cnn_lstm_model.pth"):
    """ 저장된 PyTorch 모델 불러오기 """
    model_dict = {
        "cnn_lstm": CNNLSTM(),
    }
    
    if model_name not in model_dict:
        raise ValueError(f"❌ 모델 {model_name}이 존재하지 않습니다. 가능한 모델: {list(model_dict.keys())}")

    model = model_dict[model_name]
    model.load_state_dict(torch.load(path))
    model.eval()  # 평가 모드로 변경
    print(f"✅ {model_name} 모델이 {path} 에서 불러와졌습니다.")
    return model