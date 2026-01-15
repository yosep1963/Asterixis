"""
Asterixis Detection System - PyTorch Model
1D-CNN + Bidirectional LSTM 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MODEL_CONFIG, FEATURE_DIM, WINDOW_SIZE


class AsterixisModel(nn.Module):
    """
    1D-CNN + Bidirectional LSTM 모델

    Architecture:
        Input: (batch, seq_len=15, features=127)
        Conv1D -> BatchNorm -> ReLU -> Dropout
        Conv1D -> BatchNorm -> ReLU -> Dropout
        BiLSTM -> Dropout
        FC -> ReLU -> Dropout
        FC -> Sigmoid
    """

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        seq_length=WINDOW_SIZE,
        conv1_out=MODEL_CONFIG["conv1_out"],
        conv2_out=MODEL_CONFIG["conv2_out"],
        lstm_hidden=MODEL_CONFIG["lstm_hidden"],
        fc_hidden=MODEL_CONFIG["fc_hidden"],
        dropout=MODEL_CONFIG["dropout"]
    ):
        super(AsterixisModel, self).__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length

        # 1D-CNN layers
        # Input: (batch, seq_len, input_dim) -> transpose -> (batch, input_dim, seq_len)
        self.conv1 = nn.Conv1d(input_dim, conv1_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_out)

        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_out)

        # Bidirectional LSTM
        # Input: (batch, seq_len, conv2_out)
        self.lstm = nn.LSTM(
            input_size=conv2_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        # Fully connected layers
        # BiLSTM output: lstm_hidden * 2 (bidirectional)
        self.fc1 = nn.Linear(lstm_hidden * 2, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_lstm = nn.Dropout(dropout + 0.1)  # LSTM 후 더 강한 dropout

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, seq_len, input_dim) = (batch, 15, 127)

        Returns:
            output: (batch, 1) - 확률값
        """
        # Transpose for Conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch, 127, 15)

        # Conv1D block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Conv1D block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Transpose back for LSTM: (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, 15, 128)

        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden*2)

        # 마지막 시점의 출력 사용 (또는 평균)
        # 양방향이므로 forward와 backward의 마지막 hidden state 결합
        # h_n shape: (num_layers * 2, batch, hidden)
        forward_h = h_n[-2, :, :]  # Forward 마지막 hidden
        backward_h = h_n[-1, :, :]  # Backward 마지막 hidden
        x = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden*2)

        x = self.dropout_lstm(x)

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)  # (batch, 1)

        return x.squeeze(1)  # (batch,)


class AsterixisModelLight(nn.Module):
    """
    경량 모델 (실시간 처리용)
    더 적은 파라미터로 빠른 추론
    """

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        seq_length=WINDOW_SIZE,
        hidden_dim=32
    ):
        super(AsterixisModelLight, self).__init__()

        # 단순화된 구조
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Conv1D
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        # LSTM
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # (batch, hidden)

        # FC
        x = torch.sigmoid(self.fc(x))
        return x.squeeze(1)


def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """모델 테스트"""
    # 기본 모델
    model = AsterixisModel()
    print("=" * 60)
    print("AsterixisModel (Full)")
    print("=" * 60)
    print(f"Parameters: {count_parameters(model):,}")
    print(model)

    # 테스트 입력
    batch_size = 4
    x = torch.randn(batch_size, WINDOW_SIZE, FEATURE_DIM)
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")

    # 경량 모델
    print("\n" + "=" * 60)
    print("AsterixisModelLight (Lightweight)")
    print("=" * 60)
    model_light = AsterixisModelLight()
    print(f"Parameters: {count_parameters(model_light):,}")

    with torch.no_grad():
        output_light = model_light(x)
    print(f"Output shape: {output_light.shape}")


if __name__ == "__main__":
    main()
