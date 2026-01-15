"""
Asterixis Detection System - Real-time Detector
실시간 손떨림 감지 시스템
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from pathlib import Path
import time
from datetime import datetime

from .config import (
    MODEL_DIR, CAMERA_CONFIG, MEDIAPIPE_CONFIG,
    WINDOW_SIZE, FEATURE_DIM, SEVERITY_THRESHOLDS,
    CAPTURE_DIR
)
from .model import AsterixisModel
from .preprocessor import Preprocessor


class RealtimeDetector:
    """실시간 Asterixis 감지 시스템"""

    def __init__(self, model_path=None):
        # MediaPipe 초기화
        mp_config = MEDIAPIPE_CONFIG.copy()
        mp_config["min_detection_confidence"] = 0.7
        mp_config["min_tracking_confidence"] = 0.5

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(**mp_config)
        self.mp_draw = mp.solutions.drawing_utils

        # 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)

        # 전처리기
        self.preprocessor = Preprocessor()

        # 버퍼 (0.5초 = 15 frames)
        self.coords_buffer = deque(maxlen=WINDOW_SIZE)
        self.angles_buffer = deque(maxlen=WINDOW_SIZE)

        # Severity 히스토리 (그래프용)
        self.severity_history = deque(maxlen=300)  # 10초

        # FPS 계산
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()

        # Auto-capture 설정
        self.capture_dir = CAPTURE_DIR
        self.last_capture_time = 0
        self.capture_cooldown = 3.0  # 초

        print(f"\nDevice: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def _load_model(self, model_path):
        """모델 로드"""
        if model_path is None:
            model_path = MODEL_DIR / 'asterixis_final.pt'

        model_path = Path(model_path)

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            print("Please train the model first: python -m src.train")
            return None

        model = AsterixisModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        print(f"Model loaded: {model_path}")
        return model

    def _extract_127_features(self):
        """버퍼에서 127-feature 추출"""
        coords = np.array(list(self.coords_buffer))      # (15, 42)
        angles = np.array(list(self.angles_buffer)).reshape(-1, 1)  # (15, 1)

        # 속도 계산 (1차 미분)
        velocities = np.zeros_like(coords)
        velocities[1:] = coords[1:] - coords[:-1]

        # 가속도 계산 (2차 미분)
        accelerations = np.zeros_like(coords)
        accelerations[1:] = velocities[1:] - velocities[:-1]

        # 결합
        features = np.concatenate([
            coords, angles, velocities, accelerations
        ], axis=1)  # (15, 127)

        return features

    def _calculate_severity(self, confidence, features):
        """
        Severity Score 계산 (0-100)
        - 0-25: Normal
        - 25-50: Grade 1 (Mild)
        - 50-75: Grade 2 (Moderate)
        - 75-100: Grade 3 (Severe)
        """
        base_score = confidence * 100

        # 속도 기반 가중치
        velocities = features[:, 43:85]  # 속도 부분
        velocity_magnitude = np.mean(np.abs(velocities))

        # 각도 변화량
        angles = features[:, 42]
        angle_change = np.std(angles)

        # 가중치 적용
        severity = base_score * (1 + 0.2 * velocity_magnitude * 10 + 0.1 * angle_change / 10)
        severity = min(100, max(0, severity))

        return severity

    def _get_severity_info(self, severity):
        """Severity에 따른 정보"""
        if severity < SEVERITY_THRESHOLDS["normal"]:
            return "Normal", (0, 255, 0)  # Green
        elif severity < SEVERITY_THRESHOLDS["grade1"]:
            return "Grade 1 (Mild)", (0, 255, 255)  # Yellow
        elif severity < SEVERITY_THRESHOLDS["grade2"]:
            return "Grade 2 (Moderate)", (0, 165, 255)  # Orange
        else:
            return "Grade 3 (Severe)", (0, 0, 255)  # Red

    def _auto_capture(self, frame, severity):
        """Asterixis 감지 시 자동 캡처"""
        current_time = time.time()

        if current_time - self.last_capture_time < self.capture_cooldown:
            return None

        if severity < 50:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"asterixis_severity{int(severity)}_{timestamp}.jpg"
        filepath = self.capture_dir / filename

        cv2.imwrite(str(filepath), frame)
        self.last_capture_time = current_time

        print(f"Auto-captured: {filename}")
        return filepath

    def _draw_severity_graph(self, frame):
        """Severity 히스토리 그래프"""
        if len(self.severity_history) < 2:
            return frame

        h, w = frame.shape[:2]
        graph_width = 300
        graph_height = 100
        graph_x = w - graph_width - 20
        graph_y = h - graph_height - 20

        # 배경
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (graph_x, graph_y),
                     (graph_x + graph_width, graph_y + graph_height),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.rectangle(frame,
                     (graph_x, graph_y),
                     (graph_x + graph_width, graph_y + graph_height),
                     (255, 255, 255), 1)

        # 그래프 그리기
        history = list(self.severity_history)
        points = []

        for i, severity in enumerate(history):
            x = graph_x + int((i / len(history)) * graph_width)
            y = graph_y + graph_height - int((severity / 100) * graph_height)
            points.append((x, y))

        for i in range(len(points) - 1):
            _, color = self._get_severity_info(history[i])
            cv2.line(frame, points[i], points[i+1], color, 2)

        # 레이블
        cv2.putText(frame, "Severity (10s)",
                   (graph_x + 5, graph_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def process_frame(self, frame):
        """프레임 처리 및 예측"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        prediction = None
        confidence = 0.0
        severity = 0.0
        hand_detected = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            hand_detected = True

            # 랜드마크 그리기
            self.mp_draw.draw_landmarks(
                frame, hand,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # 정규화된 좌표 추출
            normalized = self.preprocessor.normalize_landmarks(hand.landmark)
            angle = self.preprocessor.calculate_wrist_angle(hand.landmark)

            if normalized is not None:
                self.coords_buffer.append(normalized)
                self.angles_buffer.append(angle)

                if len(self.coords_buffer) == WINDOW_SIZE and self.model is not None:
                    # 127-feature 생성
                    features = self._extract_127_features()

                    # 모델 예측
                    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output = self.model(input_tensor)
                        confidence = output.item()

                    # Severity 계산
                    severity = self._calculate_severity(confidence, features)
                    self.severity_history.append(severity)

                    # 예측 결과
                    if confidence > 0.5:
                        prediction = "Asterixis: YES"
                    else:
                        prediction = "Asterixis: NO"

        return frame, prediction, confidence, severity, hand_detected

    def run(self):
        """실시간 감지 시작"""
        if self.model is None:
            print("Model not loaded. Exiting.")
            return

        cap = cv2.VideoCapture(0)

        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n" + "=" * 70)
        print("Real-time Asterixis Detection System v2.0")
        print("=" * 70)
        print("Features: 127-dim (coords + angle + velocity + acceleration)")
        print("Model: 1D-CNN + BiLSTM")
        print("-" * 70)
        print("Controls:")
        print("  Q - Quit")
        print("  S - Manual screenshot")
        print("  R - Reset history")
        print("=" * 70 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # FPS 계산
            current_time = time.time()
            fps = 1 / (current_time - self.last_time) if self.last_time else 0
            self.fps_buffer.append(fps)
            avg_fps = np.mean(self.fps_buffer)
            self.last_time = current_time

            # 프레임 처리
            frame, prediction, confidence, severity, hand_detected = self.process_frame(frame)

            # UI 그리기
            h, w = frame.shape[:2]

            # 메인 정보 패널
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            grade_text, color = self._get_severity_info(severity)
            cv2.rectangle(frame, (10, 10), (450, 200), color, 2)

            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 손 감지 상태
            hand_status = "Hand: Detected" if hand_detected else "Hand: Not Found"
            hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, hand_status,
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

            # 버퍼 상태
            buffer_status = f"Buffer: {len(self.coords_buffer)}/{WINDOW_SIZE}"
            cv2.putText(frame, buffer_status,
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 예측 결과
            if prediction:
                cv2.putText(frame, prediction,
                           (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                cv2.putText(frame, f"Severity: {severity:.1f}/100",
                           (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.putText(frame, grade_text,
                           (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Auto-capture
                self._auto_capture(frame, severity)
            else:
                cv2.putText(frame, "Analyzing...",
                           (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            # Severity 그래프
            frame = self._draw_severity_graph(frame)

            # 화면 표시
            cv2.imshow('Asterixis Detection v2.0', frame)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"manual_{timestamp}.jpg"
                filepath = self.capture_dir / filename
                cv2.imwrite(str(filepath), frame)
                print(f"Manual capture: {filename}")
            elif key == ord('r'):
                self.severity_history.clear()
                self.coords_buffer.clear()
                self.angles_buffer.clear()
                print("History reset")

        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")


def main():
    """메인 함수"""
    detector = RealtimeDetector()
    detector.run()


if __name__ == "__main__":
    main()
