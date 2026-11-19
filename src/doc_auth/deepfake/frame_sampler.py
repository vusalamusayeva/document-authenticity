from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from doc_auth.deepfake.exceptions import InvalidConfigError, VideoValidationError
from doc_auth.deepfake.video_loader import VideoStreamReader, VideoValidationConfig


@dataclass(frozen=True)
class FrameSamplingConfig:
    sampling_fps: float = 2.0
    max_frames: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.sampling_fps, bool) or not isinstance(self.sampling_fps, (int, float)):
            raise InvalidConfigError("sampling_fps rəqəm olmalıdır.")
        if float(self.sampling_fps) <= 0.0:
            raise InvalidConfigError(
                f"sampling_fps müsbət olmalıdır (sıfır və ya mənfi olmaz): {self.sampling_fps}"
            )

        if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int):
            raise InvalidConfigError("max_frames tam ədəd olmalıdır.")
        if self.max_frames <= 0:
            raise InvalidConfigError(f"max_frames müsbət tam ədəd olmalıdır: {self.max_frames}")


def sample_video_frames(
    video_path: str | Path,
    sampling_config: Optional[FrameSamplingConfig] = None,
    validation_config: Optional[VideoValidationConfig] = None,
) -> Iterator[Dict[str, object]]:
    s_cfg = sampling_config or FrameSamplingConfig()
    v_cfg = validation_config or VideoValidationConfig()

    step_interval_sec = 1.0 / float(s_cfg.sampling_fps)
    next_target_ts = 0.0
    yielded = 0
    raw_idx = 0

    try:
        with VideoStreamReader(video_path, v_cfg) as reader:
            assert reader.cap is not None
            meta = reader.metadata or {}
            native_fps = meta.get("fps") or reader.cap.get(5)
            if not native_fps or native_fps <= 0:
                native_fps = 25.0

            while yielded < s_cfg.max_frames:
                ok, frame = reader.cap.read()
                if not ok or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    break

                pos_msec = reader.cap.get(0)
                if pos_msec and pos_msec > 0:
                    timestamp = float(pos_msec) / 1000.0
                else:
                    timestamp = float(raw_idx) / float(native_fps)

                if timestamp + 1e-6 >= next_target_ts:
                    yield {
                        "frame_index": raw_idx,
                        "sample_index": yielded,
                        "timestamp": timestamp,
                        "frame": frame,
                    }
                    yielded += 1
                    next_target_ts = timestamp + step_interval_sec

                raw_idx += 1
    except VideoValidationError:
        raise
    except InvalidConfigError:
        raise
    except Exception as exc:
        raise VideoValidationError(f"Frame çıxarılması zamanı xəta baş verdi: {exc}") from exc


class FrameSampler:
    def __init__(
        self,
        sampling_fps: float = 2.0,
        max_frames: int = 32,
        validation_config: Optional[VideoValidationConfig] = None,
    ) -> None:
        self.config = FrameSamplingConfig(sampling_fps=sampling_fps, max_frames=max_frames)
        self.validation_config = validation_config or VideoValidationConfig()

    def sample_frames(self, video_path: str | Path) -> Iterator[Dict[str, object]]:
        return sample_video_frames(
            video_path=video_path,
            sampling_config=self.config,
            validation_config=self.validation_config,
        )

    def sample(self, video_path: str | Path) -> List[Dict[str, object]]:
        return list(self.sample_frames(video_path))

    def __call__(self, video_path: str | Path) -> Iterator[Dict[str, object]]:
        return self.sample_frames(video_path)