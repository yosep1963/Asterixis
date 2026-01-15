"""
Asterixis Detection System - Utilities
공통 유틸리티 함수 및 클래스
"""

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from pathlib import Path

from .config import (
    MEDIAPIPE_CONFIG, MEDIAPIPE_REALTIME_CONFIG,
    COLORS, SEVERITY_THRESHOLDS, WINDOW_SIZE
)


class MediaPipeHandler:
    """MediaPipe Hands 통합 관리 클래스"""

    def __init__(self, realtime=False):
        """
        Args:
            realtime: True면 실시간 감지용 설정 사용 (낮은 threshold)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        config = MEDIAPIPE_REALTIME_CONFIG if realtime else MEDIAPIPE_CONFIG
        self.hands = self.mp_hands.Hands(**config)

    def process(self, frame_rgb):
        """프레임에서 손 감지"""
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, hand_landmarks, color_joints=(0, 255, 0), color_connections=(0, 0, 255)):
        """랜드마크 그리기"""
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=color_joints, thickness=2, circle_radius=2),
            self.mp_draw.DrawingSpec(color=color_connections, thickness=2)
        )

    def release(self):
        """리소스 해제"""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class UIDrawer:
    """UI 그리기 유틸리티"""

    @staticmethod
    def draw_info_panel(frame, x, y, width, height, border_color=COLORS["white"]):
        """반투명 정보 패널 그리기"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), COLORS["black"], -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2)
        return frame

    @staticmethod
    def draw_text(frame, text, position, font_scale=0.7, color=COLORS["white"], thickness=2):
        """텍스트 그리기"""
        cv2.putText(
            frame, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
        )

    @staticmethod
    def draw_hand_status(frame, detected, position=(20, 70)):
        """손 감지 상태 표시"""
        status = "Hand: Detected" if detected else "Hand: Not Found"
        color = COLORS["hand_detected"] if detected else COLORS["hand_not_found"]
        UIDrawer.draw_text(frame, status, position, color=color)

    @staticmethod
    def draw_buffer_status(frame, current, total, position=(20, 100)):
        """버퍼 상태 표시"""
        text = f"Buffer: {current}/{total}"
        UIDrawer.draw_text(frame, text, position)

    @staticmethod
    def draw_fps(frame, fps, position=(20, 40)):
        """FPS 표시"""
        UIDrawer.draw_text(frame, f"FPS: {fps:.1f}", position)


class SeverityCalculator:
    """Severity Score 계산기"""

    @staticmethod
    def get_severity_info(severity):
        """Severity에 따른 Grade 텍스트와 색상 반환"""
        if severity < SEVERITY_THRESHOLDS["normal"]:
            return "Normal", COLORS["normal"]
        elif severity < SEVERITY_THRESHOLDS["grade1"]:
            return "Grade 1 (Mild)", COLORS["grade1"]
        elif severity < SEVERITY_THRESHOLDS["grade2"]:
            return "Grade 2 (Moderate)", COLORS["grade2"]
        else:
            return "Grade 3 (Severe)", COLORS["grade3"]

    @staticmethod
    def calculate_severity(confidence, features):
        """
        Severity Score 계산 (0-100)

        Args:
            confidence: 모델의 Asterixis 감지 확신도 (0-1)
            features: 127-feature 배열 (15, 127)

        Returns:
            severity: 0-100 사이의 심각도 점수
        """
        # Asterixis가 아니면 Normal
        if confidence < 0.5:
            return confidence * 50  # 0-25 범위

        # 속도 magnitude (핵심 지표)
        from .config import VELOCITY_START_IDX, VELOCITY_END_IDX, ACCEL_START_IDX, ACCEL_END_IDX

        velocities = features[:, VELOCITY_START_IDX:VELOCITY_END_IDX]
        mean_velocity = np.mean(np.abs(velocities))

        # 가속도 magnitude (보조 지표)
        accelerations = features[:, ACCEL_START_IDX:ACCEL_END_IDX]
        mean_accel = np.mean(np.abs(accelerations))

        # Grade 판별 임계값 (데이터 분석 기반)
        from .config import VELOCITY_THRESHOLDS

        grade1_min = VELOCITY_THRESHOLDS["grade1_min"]
        grade2_min = VELOCITY_THRESHOLDS["grade2_min"]
        grade3_min = VELOCITY_THRESHOLDS["grade3_min"]

        if mean_velocity < grade1_min:
            # 움직임이 작음 - Normal에 가까움
            severity = 25 + (mean_velocity / grade1_min) * 20
        elif mean_velocity < grade2_min:
            # Grade 1 범위
            ratio = (mean_velocity - grade1_min) / (grade2_min - grade1_min)
            severity = 30 + ratio * 20
        elif mean_velocity < grade3_min:
            # Grade 2 범위
            ratio = (mean_velocity - grade2_min) / (grade3_min - grade2_min)
            severity = 50 + ratio * 25
        else:
            # Grade 3 범위
            ratio = min(1.0, (mean_velocity - grade3_min) / 0.03)
            severity = 75 + ratio * 25

        # 가속도로 미세 조정 (±5)
        accel_adjust = min(5, mean_accel * 50)
        severity += accel_adjust

        return min(100, max(25, severity))


class CameraManager:
    """카메라 관리 클래스"""

    def __init__(self, camera_id=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        """프레임 읽기"""
        return self.cap.read()

    def release(self):
        """리소스 해제"""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoRecorder:
    """비디오 녹화 클래스"""

    def __init__(self, filepath, width, height, fps, codec='mp4v'):
        self.filepath = Path(filepath)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.filepath), fourcc, fps, (width, height))

    def write(self, frame):
        """프레임 쓰기"""
        self.writer.write(frame)

    def release(self):
        """리소스 해제"""
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def save_screenshot(frame, save_dir, prefix="capture"):
    """스크린샷 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = save_dir / filename

    cv2.imwrite(str(filepath), frame)
    return filepath


def create_timestamp():
    """타임스탬프 문자열 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
