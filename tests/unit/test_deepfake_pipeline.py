from pathlib import Path
import cv2
import numpy as np

from doc_auth.deepfake.exceptions import VideoValidationError
from doc_auth.deepfake.pipeline import DeepfakePipeline, analyze_video_deepfake


def _make_synthetic_video(video_path: Path, frames: int = 8) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (128, 128))
    for i in range(frames):
        frame = np.full((128, 128, 3), 180 + (i * 5) % 40, dtype=np.uint8)
        cv2.circle(frame, (64, 64), 24, (60, 80, 140), -1)
        writer.write(frame)
    writer.release()
    return video_path


def test_deepfake_pipeline_invalid_and_validation_errors(tmp_path: Path) -> None:
    pipeline = DeepfakePipeline()

    bad_res = pipeline.run(b"raw-bytes-not-a-path")
    assert bad_res["status"] == "failed"
    assert bad_res["error_type"] == "InvalidInputError"

    def _failing_validator(_path):
        raise VideoValidationError("Corrupted synthetic video")

    strict_pipeline = DeepfakePipeline(video_validator=_failing_validator)
    vid_path = _make_synthetic_video(tmp_path / "bad.mp4", frames=2)
    err_res = strict_pipeline.run(vid_path)
    assert err_res["status"] == "failed"
    assert err_res["error_type"] == "VideoValidationError"


def test_deepfake_pipeline_sample_frames_and_run_with_frames(tmp_path: Path) -> None:
    pipeline = DeepfakePipeline(video_validator=lambda _p: True)
    frame_a = np.full((128, 128, 3), 200, dtype=np.uint8)
    frame_b = np.full((128, 128, 3), 210, dtype=np.uint8)

    sampled = pipeline._sample_frames(
        tmp_path / "stub.mp4",
        [frame_a, {"frame": frame_b, "timestamp": 0.5}],
    )
    assert len(sampled) == 2
    assert sampled[0]["frame_index"] == 0
    assert sampled[1]["timestamp"] == 0.5

    run_res = pipeline.run(
        tmp_path / "stub.mp4",
        frames=[{"frame": frame_a, "frame_index": 0, "timestamp": 0.0}],
    )
    assert isinstance(run_res, dict)
    assert "status" in run_res
    assert analyze_video_deepfake is not None
