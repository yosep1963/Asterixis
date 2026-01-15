"""
Asterixis Detection System - Data Collector
Full HD 영상 데이터 수집 도구
"""

import cv2
from pathlib import Path

from .config import (
    DATA_DIR, CAMERA_CONFIG,
    RECORDING_DURATION, RECORDING_FRAMES, COLORS
)
from .utils import MediaPipeHandler, UIDrawer, CameraManager, VideoRecorder, create_timestamp


class AsterixisDataCollector:
    """Full HD 데이터 수집 도구"""

    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else DATA_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mp_handler = MediaPipeHandler(realtime=False)

    def collect_video(self, label: str, grade: int = None):
        """
        20초 비디오 수집

        Args:
            label: 'normal' or 'asterixis'
            grade: 1, 2, 3 (asterixis인 경우만)

        Returns:
            저장된 파일 경로 또는 None
        """
        with CameraManager(
            width=CAMERA_CONFIG["width"],
            height=CAMERA_CONFIG["height"],
            fps=CAMERA_CONFIG["fps"]
        ) as cam:
            # 실제 해상도 확인
            actual_width = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 파일명 생성
            filename = self._generate_filename(label, grade)
            filepath = self.save_dir / filename

            with VideoRecorder(
                filepath,
                actual_width,
                actual_height,
                CAMERA_CONFIG["fps"]
            ) as recorder:
                self._print_recording_info(filename, label, grade, actual_width, actual_height)
                frame_count = self._recording_loop(cam, recorder, actual_width, actual_height)

            return self._finalize_recording(filepath, frame_count)

    def _generate_filename(self, label, grade):
        """파일명 생성"""
        timestamp = create_timestamp()
        if label == 'normal':
            return f"normal_{timestamp}.mp4"
        return f"asterixis_grade{grade}_{timestamp}.mp4"

    def _recording_loop(self, cam, recorder, width, height):
        """녹화 루프"""
        recording = False
        frame_count = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame")
                break

            # MediaPipe 처리 (실시간 피드백)
            display_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_handler.process(frame_rgb)

            # 손 랜드마크 그리기
            hand_detected = results.multi_hand_landmarks is not None
            if hand_detected:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_handler.draw_landmarks(display_frame, hand_landmarks)

            # 녹화 처리
            if recording:
                recorder.write(frame)
                frame_count += 1

                remaining = (RECORDING_FRAMES - frame_count) / CAMERA_CONFIG["fps"]
                self._draw_recording_ui(display_frame, remaining, hand_detected)

                if frame_count >= RECORDING_FRAMES:
                    print("\nRecording complete!")
                    break
            else:
                self._draw_waiting_ui(display_frame, hand_detected)

            cv2.imshow('Data Collection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording:
                recording = True
                print("Recording started...")
            elif key == ord('q'):
                print("\nCancelled")
                break

        cv2.destroyAllWindows()
        return frame_count

    def _finalize_recording(self, filepath, frame_count):
        """녹화 완료 처리"""
        if frame_count >= RECORDING_FRAMES:
            print(f"Saved: {filepath}")
            return str(filepath)

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

    def _draw_recording_ui(self, frame, remaining, hand_detected):
        """녹화 중 UI"""
        # 배경 패널
        frame = UIDrawer.draw_info_panel(frame, 10, 10, 390, 120, COLORS["grade3"])

        # 녹화 상태
        UIDrawer.draw_text(
            frame, f"REC {remaining:.1f}s",
            (20, 50), font_scale=1.2, color=COLORS["grade3"], thickness=3
        )

        # 손 감지 상태
        UIDrawer.draw_hand_status(frame, hand_detected, (20, 100))

        return frame

    def _draw_waiting_ui(self, frame, hand_detected):
        """대기 중 UI"""
        # 배경 패널
        frame = UIDrawer.draw_info_panel(frame, 10, 10, 390, 120, COLORS["normal"])

        # 대기 상태
        UIDrawer.draw_text(
            frame, "Press SPACE to start",
            (20, 50), font_scale=0.9, color=COLORS["normal"]
        )

        # 손 감지 상태
        UIDrawer.draw_hand_status(frame, hand_detected, (20, 100))

        return frame

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

    def __del__(self):
        """리소스 해제"""
        if hasattr(self, 'mp_handler'):
            self.mp_handler.release()


def main():
    """메인 함수"""
    collector = AsterixisDataCollector()
    collector.collect_batch()


if __name__ == "__main__":
    main()
