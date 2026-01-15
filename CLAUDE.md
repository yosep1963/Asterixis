# Asterixis Detection Project

## 목적
간성뇌증 환자의 asterixis(손떨림)를 영상에서 자동 검출하는 실시간 시스템

## 기술 스택
- **언어**: Python 3.10
- **딥러닝**: PyTorch 2.7.1 + CUDA 11.8
- **손 인식**: MediaPipe Hands
- **영상 처리**: OpenCV

## 시스템 아키텍처
```
Webcam (1920×1080, 30fps)
    ↓
MediaPipe Hands (21 landmarks)
    ↓
127-Feature 추출 (정규화 좌표 + 각도 + 속도 + 가속도)
    ↓
0.5초 Sliding Window (15 frames)
    ↓
1D-CNN + BiLSTM 모델
    ↓
Asterixis YES/NO + Severity Score (0-100)
```

## 프로젝트 구조
```
src/
├── config.py          # 설정 및 하이퍼파라미터
├── data_collector.py  # 데이터 수집 도구
├── preprocessor.py    # 127-feature 추출
├── dataset.py         # PyTorch Dataset
├── model.py           # 1D-CNN + BiLSTM 모델
├── train.py           # 학습 스크립트
└── detector.py        # 실시간 Detection
```

## 주요 특징 (127차원)
1. **정규화 좌표 (42)**: 손목 중심, 손 크기로 스케일링
2. **손목 각도 (1)**: dorsiflexion 각도
3. **속도 (42)**: 프레임 간 1차 미분
4. **가속도 (42)**: 프레임 간 2차 미분

## 실행 방법
```bash
# 데이터 수집
python -m src.data_collector

# 전처리
python -m src.preprocessor

# 학습
python -m src.train

# 실시간 Detection
python -m src.detector
```

## 데이터 라벨
- `normal_*.mp4`: 정상 (Asterixis 없음)
- `asterixis_grade1_*.mp4`: Grade 1 (Mild)
- `asterixis_grade2_*.mp4`: Grade 2 (Moderate)
- `asterixis_grade3_*.mp4`: Grade 3 (Severe)
