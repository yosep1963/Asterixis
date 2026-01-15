"""
Asterixis Detection System - PyTorch Dataset
학습용 데이터셋 클래스
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

from .config import PROCESSED_DIR, TRAINING_CONFIG


class AsterixisDataset(Dataset):
    """Asterixis Detection용 PyTorch Dataset"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (N, 15, 127) 특징 배열
            y: (N,) 라벨 배열
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(data_dir=None):
    """
    전처리된 데이터 로드

    Args:
        data_dir: 데이터 디렉토리 (기본값: PROCESSED_DIR)

    Returns:
        (X, y) tuple 또는 (None, None)
    """
    data_dir = Path(data_dir) if data_dir else PROCESSED_DIR

    x_path = data_dir / 'X_train.npy'
    y_path = data_dir / 'y_train.npy'

    if not x_path.exists() or not y_path.exists():
        print(f"Data files not found in {data_dir}")
        print("Please run preprocessing first: python -m src.preprocessor")
        return None, None

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Data loaded from {data_dir}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Normal: {np.sum(y == 0):,}")
    print(f"  Asterixis: {np.sum(y == 1):,}")

    return X, y


def get_dataloaders(X, y, batch_size=None, val_split=None):
    """
    Train/Validation DataLoader 생성

    Args:
        X: 특징 배열
        y: 라벨 배열
        batch_size: 배치 크기 (기본값: config에서)
        val_split: 검증 데이터 비율 (기본값: config에서)

    Returns:
        (train_loader, val_loader, class_weights)
    """
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]
    val_split = val_split or TRAINING_CONFIG["validation_split"]

    # Train/Validation 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=42,
        stratify=y
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")

    # Dataset 생성
    train_dataset = AsterixisDataset(X_train, y_train)
    val_dataset = AsterixisDataset(X_val, y_val)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 클래스 가중치 계산 (불균형 데이터 처리)
    class_counts = np.bincount(y_train.astype(int))
    total = len(y_train)
    class_weights = torch.FloatTensor([
        total / (2 * count) for count in class_counts
    ])

    print(f"  Class weights: {class_weights.tolist()}")

    return train_loader, val_loader, class_weights


def main():
    """테스트 실행"""
    X, y = load_data()

    if X is not None:
        train_loader, val_loader, class_weights = get_dataloaders(X, y)

        # 배치 확인
        for batch_X, batch_y in train_loader:
            print(f"\nBatch shape:")
            print(f"  X: {batch_X.shape}")
            print(f"  y: {batch_y.shape}")
            break


if __name__ == "__main__":
    main()
