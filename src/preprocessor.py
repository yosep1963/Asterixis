"""
Asterixis Detection System - Preprocessor
127-Feature 추출 및 전처리 파이프라인
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .config import (
    DATA_DIR, PROCESSED_DIR,
    WINDOW_SIZE, STRIDE, FEATURE_DIM,
    NUM_LANDMARKS, COORD_DIM
)
from .utils import MediaPipeHandler


class Preprocessor:
    """127-Feature 추출 전처리기"""

    # 손 크기 최소값 (너무 작으면 무시)
    MIN_HAND_SIZE = 0.01

    def __init__(self):
        self.mp_handler = MediaPipeHandler(realtime=False)

    def normalize_landmarks(self, landmarks) -> np.ndarray:
        """
        손목 중심 정규화 + 손 크기 스케일링

        Args:
            landmarks: MediaPipe hand landmarks (21개)

        Returns:
            정규화된 42차원 벡터 [x1,y1, x2,y2, ..., x21,y21] 또는 None
        """
        if landmarks is None or len(landmarks) != NUM_LANDMARKS:
            return None

        # 손목(0번)을 원점으로
        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y

        # 손 크기 계산 (손목 -> 중지 끝)
        hand_size = self._calculate_hand_size(landmarks, wrist_x, wrist_y)

        if hand_size < self.MIN_HAND_SIZE:
            return None

        # 정규화된 좌표 계산
        normalized = []
        for lm in landmarks:
            norm_x = (lm.x - wrist_x) / hand_size
            norm_y = (lm.y - wrist_y) / hand_size
            normalized.extend([norm_x, norm_y])

        return np.array(normalized)  # Shape: (42,)

    def _calculate_hand_size(self, landmarks, wrist_x, wrist_y):
        """손 크기 계산 (손목 -> 중지 끝)"""
        middle_tip_x = landmarks[12].x
        middle_tip_y = landmarks[12].y
        return np.sqrt(
            (middle_tip_x - wrist_x) ** 2 +
            (middle_tip_y - wrist_y) ** 2
        )

    def calculate_wrist_angle(self, landmarks) -> float:
        """
        손목 각도 계산 (dorsiflexion 각도)

        Args:
            landmarks: MediaPipe hand landmarks

        Returns:
            각도 (degrees)
        """
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y])
        middle_mcp = np.array([landmarks[9].x, landmarks[9].y])

        # 손바닥 중심 방향
        palm_center = (index_mcp + middle_mcp) / 2
        palm_vector = palm_center - wrist

        # 수직선(y축)과의 각도
        angle_rad = np.arctan2(palm_vector[0], -palm_vector[1])
        return np.degrees(angle_rad)

    def extract_features_from_video(self, video_path: str) -> np.ndarray:
        """
        비디오에서 127-feature 추출

        Args:
            video_path: 비디오 파일 경로

        Returns:
            (frames, 127) numpy array 또는 None
        """
        cap = cv2.VideoCapture(video_path)

        coords_list = []
        angles_list = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self._process_frame(frame, coords_list, angles_list)
        finally:
            cap.release()

        if len(coords_list) == 0:
            return None

        return self._build_features(coords_list, angles_list)

    def _process_frame(self, frame, coords_list, angles_list):
        """단일 프레임 처리"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_handler.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0].landmark
            normalized = self.normalize_landmarks(hand)
            angle = self.calculate_wrist_angle(hand)

            if normalized is not None:
                coords_list.append(normalized)
                angles_list.append(angle)
            elif coords_list:
                # 정규화 실패 시 이전 값 복사
                coords_list.append(coords_list[-1])
                angles_list.append(angles_list[-1])
        elif coords_list:
            # 손 없으면 이전 값 복사
            coords_list.append(coords_list[-1])
            angles_list.append(angles_list[-1])

    def _build_features(self, coords_list, angles_list):
        """좌표와 각도에서 127-feature 배열 생성"""
        coords = np.array(coords_list)
        angles = np.array(angles_list).reshape(-1, 1)

        # 속도 계산 (1차 미분)
        velocities = np.zeros_like(coords)
        velocities[1:] = coords[1:] - coords[:-1]

        # 가속도 계산 (2차 미분)
        accelerations = np.zeros_like(coords)
        accelerations[1:] = velocities[1:] - velocities[:-1]

        # 모든 특징 결합: 42 + 1 + 42 + 42 = 127
        return np.concatenate([
            coords,         # 42: 정규화된 x,y 좌표
            angles,         # 1: 손목 각도
            velocities,     # 42: 속도
            accelerations   # 42: 가속도
        ], axis=1)

    def create_windows(self, features: np.ndarray) -> np.ndarray:
        """
        Sliding window 생성

        Args:
            features: (frames, 127) 배열

        Returns:
            (windows, WINDOW_SIZE, 127) 배열
        """
        windows = []

        for i in range(0, len(features) - WINDOW_SIZE + 1, STRIDE):
            window = features[i:i + WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                windows.append(window)

        return np.array(windows)

    def process_dataset(self, dataset_dir=None, output_dir=None):
        """
        전체 데이터셋 처리

        Args:
            dataset_dir: 비디오 파일 디렉토리
            output_dir: 출력 디렉토리

        Returns:
            (X, y) tuple
        """
        dataset_dir = Path(dataset_dir) if dataset_dir else DATA_DIR
        output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        self._print_header(dataset_dir, output_dir)

        all_windows, all_labels = self._process_all_videos(dataset_dir)

        if len(all_windows) == 0:
            print("\nNo data found. Please collect data first.")
            return None, None

        X, y = self._save_and_report(all_windows, all_labels, output_dir)
        return X, y

    def _print_header(self, dataset_dir, output_dir):
        """처리 헤더 출력"""
        print("\n" + "=" * 70)
        print("127-Feature Extraction Pipeline")
        print("=" * 70)
        print(f"Input directory: {dataset_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Window size: {WINDOW_SIZE} frames (0.5 sec)")
        print(f"Stride: {STRIDE} frames")
        print(f"Feature dimension: {FEATURE_DIM}")
        print("=" * 70)

    def _process_all_videos(self, dataset_dir):
        """모든 비디오 처리"""
        all_windows = []
        all_labels = []

        # Normal 비디오 처리
        normal_videos = list(dataset_dir.glob('normal_*.mp4'))
        print(f"\nProcessing {len(normal_videos)} Normal videos...")

        for video_path in tqdm(normal_videos, desc="Normal"):
            self._process_single_video(video_path, 0, all_windows, all_labels)

        # Asterixis 비디오 처리 (Grade 1, 2, 3)
        for grade in [1, 2, 3]:
            asterixis_videos = list(dataset_dir.glob(f'asterixis_grade{grade}_*.mp4'))
            print(f"\nProcessing {len(asterixis_videos)} Grade {grade} videos...")

            for video_path in tqdm(asterixis_videos, desc=f"Grade {grade}"):
                self._process_single_video(video_path, 1, all_windows, all_labels)

        return all_windows, all_labels

    def _process_single_video(self, video_path, label, all_windows, all_labels):
        """단일 비디오 처리 및 결과 추가"""
        features = self.extract_features_from_video(str(video_path))

        if features is not None:
            windows = self.create_windows(features)
            all_windows.extend(windows)
            all_labels.extend([label] * len(windows))

    def _save_and_report(self, all_windows, all_labels, output_dir):
        """결과 저장 및 통계 출력"""
        X = np.array(all_windows, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        # 저장
        np.save(output_dir / 'X_train.npy', X)
        np.save(output_dir / 'y_train.npy', y)

        # 통계 출력
        print("\n" + "=" * 70)
        print("Processing Complete!")
        print("=" * 70)
        print(f"Total windows: {len(X):,}")
        print(f"  - Normal (0): {np.sum(y == 0):,}")
        print(f"  - Asterixis (1): {np.sum(y == 1):,}")
        print(f"Data shape: {X.shape}")
        print(f"Feature breakdown:")
        print(f"  - Normalized coords: {COORD_DIM} (21 landmarks x 2)")
        print(f"  - Wrist angle: 1")
        print(f"  - Velocity: {COORD_DIM}")
        print(f"  - Acceleration: {COORD_DIM}")
        print(f"  - Total: {FEATURE_DIM}")
        print(f"\nSaved files:")
        print(f"  - {output_dir / 'X_train.npy'}")
        print(f"  - {output_dir / 'y_train.npy'}")
        print("=" * 70)

        return X, y

    def __del__(self):
        """리소스 해제"""
        if hasattr(self, 'mp_handler'):
            self.mp_handler.release()


def main():
    """메인 함수"""
    preprocessor = Preprocessor()
    X, y = preprocessor.process_dataset()

    if X is not None:
        print(f"\nData ready for training!")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()
