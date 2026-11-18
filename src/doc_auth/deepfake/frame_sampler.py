import cv2
import numpy as np
from typing import List, Dict, Any, Generator
from doc_auth.deepfake.exceptions import VideoLoadError

def sample_frames(video_path: str, sample_rate: int = 1, max_frames: int = 100) -> Generator[Dict[str, Any], None, None]:
    if sample_rate <= 0:
        raise ValueError("Sampling rate 0-dan böyük olmalıdır.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoLoadError(f"Video decoder açılmadı: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        frame_count = 0
        extracted_count = 0
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_rate == 0:
                timestamp_ms = int((frame_count / fps) * 1000)
                yield {
                    "frame_index": frame_count,
                    "timestamp_ms": timestamp_ms,
                    "frame": frame
                }
                extracted_count += 1
            frame_count += 1
    finally:
        cap.release()

class FrameSampler:
    def __init__(self, sample_rate: int = 1, max_frames: int = 100):
        self.sample_rate = sample_rate
        self.max_frames = max_frames

    def sample_frames(self, video_path: str) -> Generator[Dict[str, Any], None, None]:
        return sample_frames(video_path, self.sample_rate, self.max_frames)

    def sample(self, video_path: str) -> List[Dict[str, Any]]:
        return list(sample_frames(video_path, self.sample_rate, self.max_frames))