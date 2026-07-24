"""
EAG-GRF Deep Learning Models

Model 1: EAG1DCNN — 경량 1D-CNN (CPU 프로토타입용, ~50K params)
Model 2: EAGLSTM — CNN + BiLSTM (GPU용)
Model 3: EAGRegressor — GRF 재구성 (Task C)
"""

import torch
import torch.nn as nn


class EAG1DCNN(nn.Module):
    """경량 1D-CNN 분류기.

    Input: (batch, 8, 3750)
    Output: (batch, n_classes)
    """

    def __init__(self, n_channels: int = 8, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 64, kernel_size=4, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)       # (B, 64, 1)
        x = x.squeeze(-1)          # (B, 64)
        return self.classifier(x)   # (B, n_classes)


class EAGLSTM(nn.Module):
    """CNN + BiLSTM 분류기 (GPU용).

    Input: (batch, 8, 3750)
    Output: (batch, n_classes)
    """

    def __init__(self, n_channels: int = 8, n_classes: int = 3,
                 hidden_size: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 2, n_classes),
        )

    def forward(self, x):
        x = self.cnn(x)                    # (B, 64, T')
        x = x.permute(0, 2, 1)             # (B, T', 64)
        _, (h, _) = self.lstm(x)            # h: (2, B, hidden)
        h = torch.cat([h[0], h[1]], dim=1)  # (B, hidden*2)
        return self.classifier(h)


class EAGRegressor(nn.Module):
    """EAG → GRF 재구성 (Task C).

    Input: (batch, 8, 3750)
    Output: (batch, 2, 3750) — left_grf, right_grf 예측
    """

    def __init__(self, n_channels: int = 8, n_out: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=16, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.decoder = nn.Conv1d(32, n_out, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)     # (B, 32, T) — stride=1이므로 길이 유지
        out = self.decoder(feat)   # (B, 2, T)
        # 입력과 출력 길이 맞추기
        if out.shape[-1] != x.shape[-1]:
            out = nn.functional.interpolate(out, size=x.shape[-1], mode='linear',
                                            align_corners=False)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 모델 크기 확인
    for name, Model, kwargs in [
        ('EAG1DCNN', EAG1DCNN, {'n_classes': 3}),
        ('EAGLSTM', EAGLSTM, {'n_classes': 3}),
        ('EAGRegressor', EAGRegressor, {}),
    ]:
        m = Model(**kwargs)
        x = torch.randn(2, 8, 3750)
        y = m(x)
        print(f"{name}: params={count_params(m):,}, "
              f"input={x.shape}, output={y.shape}")
