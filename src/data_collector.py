"""
Asterixis Detection System - Data Collector
Full HD 영상 데이터 수집 도구
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from pathlib import Path

from .config import (
    DATA_DIR, CAMERA_CONFIG, MEDIAPIPE_CONFIG,
    RECORDING_DURATION, RECORDING_FRAMES
)


class AsterixisDataCollector:
    """Full HD 데이터 수집 도구"""

    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else DATA_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG)
        self.mp_draw = mp.solutions.drawing_utils

    def collect_video(self, label: str, grade: int = None):
        """
        20초 비디오 수집

        Args:
            label: 'normal' or 'asterixis'
            grade: 1, 2, 3 (asterixis인 경우만)

        Returns:
            저장된 파일 경로 또는 None
        """
        cap = cv2.VideoCapture(0)

        # Full HD 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["height"])
        cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])

        # 실제 해상도 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if label == 'normal':
            filename = f"normal_{timestamp}.mp4"
        else:
            filename = f"asterixis_grade{grade}_{timestamp}.mp4"

        filepath = self.save_dir / filename

        # VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(filepath), fourcc,
            float(CAMERA_CONFIG["fps"]),
            (actual_width, actual_height)
        )

        self._print_recording_info(filename, label, grade, actual_width, actual_height)

        recording = False
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # MediaPipe 처리 (실시간 피드백)
            display_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # 손 랜드마크 그리기
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        display_frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

            # 녹화 중이면 원본 프레임 저장 (랜드마크 없이)
            if recording:
                out.write(frame)
                frame_count += 1

                # 진행 상태 표시
                remaining = (RECORDING_FRAMES - frame_count) / CAMERA_CONFIG["fps"]
                self._draw_recording_ui(display_frame, remaining, results)

                if frame_count >= RECORDING_FRAMES:
                    print("\nRecording complete!")
                    break
            else:
                self._draw_waiting_ui(display_frame, results)

            cv2.imshow('Data Collection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording:
                recording = True
                print("Recording started...")
            elif key == ord('q'):
                print("\nCancelled")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if frame_count >= RECORDING_FRAMES:
            print(f"Saved: {filepath}")
            return str(filepath)
        else:
            # 취소된 경우 파일 삭제
            if filepath.exists():
                filepath.unlink()
            return None

    def _print_recording_info(self, filename, label, grade, width, height):
        """녹화 정보 출력"""
        print("\n" + "=" * 60)
        print(f"Recording: {filename}")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {RECORDING_DURATION} seconds")
        print(f"Label: {label}" + (f" (Grade {grade})" if grade else ""))
        print("-" * 60)
        print("Press SPACE to start recording")
        print("Press Q to quit")
        print("=" * 60 + "\n")

    def _draw_recording_ui(self, frame, remaining, results):
        """녹화 중 UI"""
        h, w = frame.shape[:2]

        # 배경 박스
        cv2.rectangle(frame, (10, 10), (400, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 130), (0, 0, 255), 2)

        # 녹화 상태
        cv2.putText(frame, f"REC {remaining:.1f}s",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 손 감지 상태
        hand_status = "Hand: Detected" if results.multi_hand_landmarks else "Hand: Not Found"
        hand_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
        cv2.putText(frame, hand_status,
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)

    def _draw_waiting_ui(self, frame, results):
        """대기 중 UI"""
        h, w = frame.shape[:2]

        # 배경 박스
        cv2.rectangle(frame, (10, 10), (400, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 130), (0, 255, 0), 2)

        # 대기 상태
        cv2.putText(frame, "Press SPACE to start",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 손 감지 상태
        hand_status = "Hand: Detected" if results.multi_hand_landmarks else "Hand: Not Found"
        hand_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
        cv2.putText(frame, hand_status,
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)

    def collect_batch(self):
        """대화형 배치 수집"""
        print("\n" + "=" * 60)
        print("Asterixis Data Collection Tool")
        print("=" * 60)
        print("\nData Collection Guide:")
        print("1. Normal: Hold hand steady without asterixis for 20 seconds")
        print("2. Grade 1: Perform mild asterixis continuously")
        print("3. Grade 2: Perform moderate asterixis continuously")
        print("4. Grade 3: Perform severe asterixis continuously")
        print("\nCommands:")
        print("  n - Collect Normal sample")
        print("  1 - Collect Grade 1 sample")
        print("  2 - Collect Grade 2 sample")
        print("  3 - Collect Grade 3 sample")
        print("  q - Quit")
        print("=" * 60)

        while True:
            cmd = input("\nEnter command (n/1/2/3/q): ").strip().lower()

            if cmd == 'q':
                print("Exiting data collection.")
                break
            elif cmd == 'n':
                self.collect_video('normal')
            elif cmd in ['1', '2', '3']:
                self.collect_video('asterixis', grade=int(cmd))
            else:
                print("Invalid command. Use n, 1, 2, 3, or q.")


def main():
    """메인 함수"""
    collector = AsterixisDataCollector()
    collector.collect_batch()


if __name__ == "__main__":
    main()
