import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from doc_auth.deepfake.exceptions import InvalidConfigError, VideoLoadError, VideoValidationError


@dataclass
class VideoLoaderConfig:
    probe_timeout_sec: float = 10.0
    max_width: Optional[int] = None
    max_height: Optional[int] = None


VideoValidationConfig = VideoLoaderConfig


def probe_video_metadata(video_path: str | Path, timeout_sec: float = 10.0) -> Dict[str, Any]:
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise VideoLoadError(f"Video faylı tapılmadı: {path}")
    if path.stat().st_size == 0:
        raise VideoLoadError(f"Video faylı boşdur: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VideoLoadError(f"ffprobe tapılmadı: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoLoadError(f"ffprobe vaxt aşımına uğradı: {path}") from exc

    if proc.returncode != 0 or not proc.stdout:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise VideoLoadError(f"Video faylı açıla bilmədi: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if width <= 0 or height <= 0:
            raise VideoValidationError(f"Video ölçüləri etibarsızdır: {width}x{height}")
        duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
        return {
            "width": width,
            "height": height,
            "fps": fps if fps > 0 else 25.0,
            "frame_count": frame_count,
            "duration": duration,
        }

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoLoadError(f"ffprobe çıxışı oxuna bilmədi: {exc}") from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise VideoValidationError(f"Video axını (stream) tapılmadı: {path}")

    width = int(v_stream.get("width") or 0)
    height = int(v_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoValidationError(f"Etibarsız video ölçüləri: {width}x{height}")

    fps_raw = str(v_stream.get("r_frame_rate") or "25/1")
    fps = 25.0
    if "/" in fps_raw:
        num_s, den_s = fps_raw.split("/", 1)
        try:
            den_v = float(den_s)
            if den_v > 0:
                fps = float(num_s) / den_v
        except ValueError:
            fps = 25.0
    else:
        try:
            fps = float(fps_raw)
        except ValueError:
            fps = 25.0

    duration = 0.0
    dur_raw = v_stream.get("duration") or fmt.get("duration")
    if dur_raw is not None:
        try:
            duration = max(0.0, float(dur_raw))
        except ValueError:
            duration = 0.0

    frame_count = 0
    nb_raw = v_stream.get("nb_frames")
    if nb_raw is not None:
        try:
            frame_count = max(0, int(nb_raw))
        except ValueError:
            frame_count = 0
    if frame_count == 0 and duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))

    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 25.0,
        "frame_count": frame_count,
        "duration": duration,
    }


def validate_video(
    video_path: str | Path,
    config: Optional[VideoLoaderConfig] = None,
) -> Dict[str, Any]:
    cfg = config or VideoLoaderConfig()
    if cfg.probe_timeout_sec <= 0:
        raise InvalidConfigError("probe_timeout_sec müsbət olmalıdır.")
    meta = probe_video_metadata(video_path, timeout_sec=cfg.probe_timeout_sec)
    if cfg.max_width is not None and meta["width"] > cfg.max_width:
        raise VideoValidationError(f"Video eni ({meta['width']}) limiti ({cfg.max_width}) aşır.")
    if cfg.max_height is not None and meta["height"] > cfg.max_height:
        raise VideoValidationError(f"Video hündürlüyü ({meta['height']}) limiti ({cfg.max_height}) aşır.")
    return meta


class VideoStreamReader:
    def __init__(
        self,
        video_path: str | Path,
        config: Optional[VideoLoaderConfig] = None,
    ) -> None:
        self.path = Path(video_path)
        self.config = config or VideoLoaderConfig()
        self.metadata: Dict[str, Any] = {}
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoStreamReader":
        self.metadata = validate_video(self.path, self.config)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise VideoLoadError(f"Video kadr oxuyucusu açıla bilmədi: {self.path}")
        self._cap = cap
        return self

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise VideoLoadError("VideoStreamReader konteksti açılmayıb.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None