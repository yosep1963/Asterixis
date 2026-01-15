# Asterixis Detection System v2.0

간성뇌증 환자의 Asterixis(손떨림)를 실시간으로 감지하는 딥러닝 시스템

## 시스템 아키텍처

```
Webcam (1920x1080, 30fps)
    ↓
MediaPipe Hands (21 landmarks)
    ↓
127-Feature 추출
  - 정규화 좌표 (42)
  - 손목 각도 (1)
  - 속도 (42)
  - 가속도 (42)
    ↓
0.5초 Sliding Window (15 frames)
    ↓
1D-CNN + BiLSTM (PyTorch)
    ↓
Asterixis YES/NO + Severity Score (0-100)
```

## 프로젝트 구조

```
asterixis-detection/
├── data/
│   └── asterixis_dataset/    # 수집된 mp4 파일
├── src/
│   ├── __init__.py
│   ├── config.py             # 설정 및 하이퍼파라미터
│   ├── data_collector.py     # 데이터 수집 도구
│   ├── preprocessor.py       # 127-feature 추출
│   ├── dataset.py            # PyTorch Dataset
│   ├── model.py              # 1D-CNN + BiLSTM 모델
│   ├── train.py              # 학습 스크립트
│   └── detector.py           # 실시간 Detection
├── models/                   # 학습된 모델
├── processed_data/           # 전처리된 데이터
├── captures/                 # 자동 캡처 이미지
├── notebooks/                # 실험용 노트북
├── CLAUDE.md                 # 프로젝트 컨텍스트
└── README.md
```

## 설치 방법

```bash
# Conda 환경 활성화
conda activate asterixis

# 프로젝트 디렉토리 이동
cd ~/asterixis-detection
```

## 사용 방법

### 1. 데이터 수집

```bash
python -m src.data_collector
```

- `n`: Normal 샘플 수집 (20초)
- `1`: Grade 1 Asterixis 수집
- `2`: Grade 2 Asterixis 수집
- `3`: Grade 3 Asterixis 수집
- `q`: 종료

### 2. 데이터 전처리

```bash
python -m src.preprocessor
```

- 수집된 비디오에서 127-feature 추출
- Sliding window 생성
- `processed_data/` 디렉토리에 저장

### 3. 모델 학습

```bash
python -m src.train
```

- 1D-CNN + BiLSTM 모델 학습
- Early stopping 적용
- TensorBoard 로깅 (`runs/` 디렉토리)
- 최적 모델 `models/asterixis_final.pt` 저장

### 4. 실시간 Detection

```bash
python -m src.detector
```

**키보드 조작:**
- `Q`: 종료
- `S`: 수동 스크린샷
- `R`: 히스토리 초기화

## 기술 스택

- **Python** 3.10
- **PyTorch** 2.7.1 + CUDA 11.8
- **MediaPipe** 0.10.31
- **OpenCV** 4.12.0

## 성능 목표

| 지표 | 목표값 |
|------|--------|
| Detection Accuracy | ≥ 90% |
| Sensitivity (Grade 2-3) | ≥ 95% |
| Specificity | ≥ 85% |
| Processing Speed | ≥ 25 FPS |

## Severity Score 기준

- **0-25**: Normal (정상)
- **25-50**: Grade 1 (Mild)
- **50-75**: Grade 2 (Moderate)
- **75-100**: Grade 3 (Severe)

## 라이선스

Medical Research Use Only
