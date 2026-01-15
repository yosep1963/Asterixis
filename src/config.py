"""
Asterixis Detection System - Configuration
설정 파일: 하이퍼파라미터, 경로, 상수 정의
"""

import os
from pathlib import Path

# =============================================================================
# 프로젝트 경로
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "asterixis_dataset"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
MODEL_DIR = PROJECT_ROOT / "models"
CAPTURE_DIR = PROJECT_ROOT / "captures"

# 디렉토리 자동 생성
for dir_path in [DATA_DIR, PROCESSED_DIR, MODEL_DIR, CAPTURE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MediaPipe 설정
# =============================================================================
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.8,
    "min_tracking_confidence": 0.7,
}

# =============================================================================
# 카메라 설정
# =============================================================================
CAMERA_CONFIG = {
    "width": 1920,          # Full HD
    "height": 1080,
    "fps": 30,
}

# =============================================================================
# 특징 추출 설정
# =============================================================================
NUM_LANDMARKS = 21          # MediaPipe 손 랜드마크 개수
COORD_DIM = 42              # 21 landmarks × 2 (x, y)
ANGLE_DIM = 1               # 손목 각도
VELOCITY_DIM = 42           # 속도 (42차원)
ACCELERATION_DIM = 42       # 가속도 (42차원)
FEATURE_DIM = COORD_DIM + ANGLE_DIM + VELOCITY_DIM + ACCELERATION_DIM  # 127

# =============================================================================
# 윈도우 설정
# =============================================================================
WINDOW_SIZE = 15            # 0.5초 (30fps 기준)
STRIDE = 3                  # 0.1초 간격으로 슬라이딩

# =============================================================================
# 데이터 수집 설정
# =============================================================================
RECORDING_DURATION = 20     # 초
RECORDING_FRAMES = RECORDING_DURATION * CAMERA_CONFIG["fps"]  # 600 frames

# =============================================================================
# 모델 하이퍼파라미터
# =============================================================================
MODEL_CONFIG = {
    "input_dim": FEATURE_DIM,       # 127
    "seq_length": WINDOW_SIZE,      # 15
    "conv1_out": 64,
    "conv2_out": 128,
    "lstm_hidden": 64,
    "fc_hidden": 32,
    "dropout": 0.3,
}

# =============================================================================
# 학습 설정
# =============================================================================
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
}

# =============================================================================
# Severity Score 임계값
# =============================================================================
SEVERITY_THRESHOLDS = {
    "normal": 25,           # 0-25: Normal
    "grade1": 50,           # 25-50: Grade 1 (Mild)
    "grade2": 75,           # 50-75: Grade 2 (Moderate)
    "grade3": 100,          # 75-100: Grade 3 (Severe)
}

# =============================================================================
# 라벨 정의
# =============================================================================
LABELS = {
    0: "Normal",
    1: "Asterixis",
}

GRADE_LABELS = {
    0: "Normal",
    1: "Grade 1 (Mild)",
    2: "Grade 2 (Moderate)",
    3: "Grade 3 (Severe)",
}
