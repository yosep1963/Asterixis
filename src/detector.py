"""
Asterixis Detection System - Real-time Detector
실시간 손떨림 감지 시스템
"""

import cv2
import numpy as np
import torch
from collections import deque
from pathlib import Path
import time

from .config import (
    MODEL_DIR, CAMERA_REALTIME_CONFIG, WINDOW_SIZE,
    CAPTURE_DIR, DETECTION_CONFIG, COLORS,
    VELOCITY_START_IDX, VELOCITY_END_IDX
)
from .model import AsterixisModel
from .preprocessor import Preprocessor
from .utils import (
    MediaPipeHandler, UIDrawer, CameraManager,
    SeverityCalculator, save_screenshot, create_timestamp
)


class RealtimeDetector:
    """실시간 Asterixis 감지 시스템"""

    def __init__(self, model_path=None):
        # MediaPipe 초기화 (실시간용 설정)
        self.mp_handler = MediaPipeHandler(realtime=True)

        # 모델 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)

        # 전처리기
        self.preprocessor = Preprocessor()

        # 버퍼 (0.5초 = 15 frames)
        self.coords_buffer = deque(maxlen=WINDOW_SIZE)
        self.angles_buffer = deque(maxlen=WINDOW_SIZE)

        # Severity 히스토리 (그래프용)
        self.severity_history = deque(maxlen=DETECTION_CONFIG["severity_history_size"])

        # FPS 계산
        self.fps_buffer = deque(maxlen=DETECTION_CONFIG["fps_history_size"])
        self.last_time = time.time()

        # Auto-capture 설정
        self.capture_dir = CAPTURE_DIR
        self.last_capture_time = 0

        self._print_device_info()

    def _print_device_info(self):
        """디바이스 정보 출력"""
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
        coords = np.array(list(self.coords_buffer))
        angles = np.array(list(self.angles_buffer)).reshape(-1, 1)

        # 속도 계산 (1차 미분)
        velocities = np.zeros_like(coords)
        velocities[1:] = coords[1:] - coords[:-1]

        # 가속도 계산 (2차 미분)
        accelerations = np.zeros_like(coords)
        accelerations[1:] = velocities[1:] - velocities[:-1]

        # 결합
        return np.concatenate([
            coords, angles, velocities, accelerations
        ], axis=1)

    def _auto_capture(self, frame, severity):
        """Asterixis 감지 시 자동 캡처"""
        current_time = time.time()
        cooldown = DETECTION_CONFIG["auto_capture_cooldown"]
        threshold = DETECTION_CONFIG["auto_capture_threshold"]

        if current_time - self.last_capture_time < cooldown:
            return None

        if severity < threshold:
            return None

        filepath = save_screenshot(
            frame, self.capture_dir,
            prefix=f"asterixis_severity{int(severity)}"
        )
        self.last_capture_time = current_time
        print(f"Auto-captured: {filepath.name}")
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

        # 배경 패널
        frame = UIDrawer.draw_info_panel(
            frame, graph_x, graph_y, graph_width, graph_height
        )

        # 그래프 그리기
        history = list(self.severity_history)
        points = []

        for i, severity in enumerate(history):
            x = graph_x + int((i / len(history)) * graph_width)
            y = graph_y + graph_height - int((severity / 100) * graph_height)
            points.append((x, y))

        for i in range(len(points) - 1):
            _, color = SeverityCalculator.get_severity_info(history[i])
            cv2.line(frame, points[i], points[i+1], color, 2)

        # 레이블
        UIDrawer.draw_text(
            frame, "Severity (10s)",
            (graph_x + 5, graph_y - 5), font_scale=0.5
        )

        return frame

    def process_frame(self, frame):
        """프레임 처리 및 예측"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_handler.process(frame_rgb)

        prediction = None
        confidence = 0.0
        severity = 0.0
        hand_detected = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            hand_detected = True

            # 랜드마크 그리기
            self.mp_handler.draw_landmarks(frame, hand)

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
                    severity = SeverityCalculator.calculate_severity(confidence, features)
                    self.severity_history.append(severity)

                    # 예측 결과
                    prediction = "Asterixis: YES" if confidence > 0.5 else "Asterixis: NO"

        return frame, prediction, confidence, severity, hand_detected

    def _draw_main_ui(self, frame, prediction, severity, hand_detected, avg_fps):
        """메인 UI 그리기"""
        grade_text, color = SeverityCalculator.get_severity_info(severity)

        # 메인 정보 패널
        frame = UIDrawer.draw_info_panel(frame, 10, 10, 440, 190, color)

        # FPS
        UIDrawer.draw_fps(frame, avg_fps)

        # 손 감지 상태
        UIDrawer.draw_hand_status(frame, hand_detected)

        # 버퍼 상태
        UIDrawer.draw_buffer_status(frame, len(self.coords_buffer), WINDOW_SIZE)

        # 예측 결과
        if prediction:
            UIDrawer.draw_text(frame, prediction, (20, 140), font_scale=1.0, color=color)
            UIDrawer.draw_text(
                frame, f"Severity: {severity:.1f}/100",
                (20, 170), font_scale=0.8, color=color
            )
            UIDrawer.draw_text(frame, grade_text, (20, 195), font_scale=0.6, color=color)
        else:
            UIDrawer.draw_text(
                frame, "Analyzing...",
                (20, 140), font_scale=1.0, color=COLORS["grade1"]
            )

        return frame

    def run(self):
        """실시간 감지 시작"""
        if self.model is None:
            print("Model not loaded. Exiting.")
            return

        with CameraManager(
            width=CAMERA_REALTIME_CONFIG["width"],
            height=CAMERA_REALTIME_CONFIG["height"],
            fps=CAMERA_REALTIME_CONFIG["fps"]
        ) as cam:
            self._print_controls()

            while True:
                ret, frame = cam.read()
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
                frame = self._draw_main_ui(frame, prediction, severity, hand_detected, avg_fps)

                # Auto-capture
                if prediction:
                    self._auto_capture(frame, severity)

                # Severity 그래프
                frame = self._draw_severity_graph(frame)

                # 화면 표시
                cv2.imshow('Asterixis Detection v2.0', frame)

                # 키 입력 처리
                if self._handle_key_input(frame):
                    break

        cv2.destroyAllWindows()
        print("Detection stopped")

    def _print_controls(self):
        """컨트롤 안내 출력"""
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

    def _handle_key_input(self, frame):
        """키 입력 처리. 종료 시 True 반환"""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nExiting...")
            return True
        elif key == ord('s'):
            filepath = save_screenshot(frame, self.capture_dir, prefix="manual")
            print(f"Manual capture: {filepath.name}")
        elif key == ord('r'):
            self.severity_history.clear()
            self.coords_buffer.clear()
            self.angles_buffer.clear()
            print("History reset")

        return False

    def __del__(self):
        """리소스 해제"""
        if hasattr(self, 'mp_handler'):
            self.mp_handler.release()


def main():
    """메인 함수"""
    detector = RealtimeDetector()
    detector.run()


if __name__ == "__main__":
    main()
