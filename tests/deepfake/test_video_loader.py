from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_auth.deepfake.exceptions import VideoLoadError, VideoValidationError
from doc_auth.deepfake.video_loader import VideoStreamReader, validate_video


def test_validate_video_missing_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(VideoLoadError):
        validate_video(tmp_path / "missing.mp4")

    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(VideoLoadError):
        validate_video(empty_file)


def test_video_stream_reader_synthetic_avi(tmp_path: Path) -> None:
    video_file = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_file), fourcc, 10.0, (160, 120))
    for _ in range(5):
        frame = np.full((120, 160, 3), 120, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = []
    with VideoStreamReader(video_file) as reader:
        while True:
            f = reader.read_frame()
            if f is None:
                break
            frames.append(f)
    assert len(frames) >= 1