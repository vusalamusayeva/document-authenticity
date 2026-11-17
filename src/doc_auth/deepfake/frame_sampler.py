import cv2
import os
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from doc_auth.schemas.pipeline_schemas import DeepfakeConfig
from doc_auth.deepfake.exceptions import VideoLoadError, SamplingLimitError

class FrameSampler:
    def __init__(self, config: DeepfakeConfig = None):
        self.config = config or DeepfakeConfig()
        self.temp_dir = None

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
            self.temp_dir = None

    def sample_frames(self, video_path: str) -> Generator[Dict[str, Any], None, None]:
        path = Path(video_path)
        if not path.exists():
            raise VideoLoadError(f"Video faylı tapılmadı: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoLoadError(f"Video açılmadı: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        interval = max(1, int(fps / self.config.sampling_rate_fps))
        frames_extracted = 0
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    if frames_extracted >= self.config.max_frames:
                        raise SamplingLimitError("Frame limit aşdı")

                    timestamp_ms = (frame_idx / fps) * 1000.0
                    yield {
                        "frame_index": frame_idx,
                        "timestamp_ms": round(timestamp_ms, 2),
                        "frame": frame
                    }
                    frames_extracted += 1
                frame_idx += 1
        finally:
            cap.release()