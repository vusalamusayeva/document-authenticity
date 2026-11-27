from pathlib import Path
import cv2
import numpy as np

from doc_auth.deepfake.exceptions import ModelUnavailableError, VideoValidationError
from doc_auth.deepfake.pipeline import DeepfakePipeline, analyze_video_deepfake


class _StubAudio:
    def analyze(self, _path):
        return {"status": "ok", "lip_sync_score": 0.9}


class _StubFaceDetectorNoFace:
    def detect_faces(self, _frame):
        return []


class _StubFaceDetectorWithFace:
    def detect_faces(self, _frame):
        return [{"bbox": [10, 10, 60, 60], "score": 0.99}]


class _StubVisualDetector:
    threshold = 0.5

    def __init__(self, fail_weights: bool = False) -> None:
        self.fail_weights = fail_weights

    def verify_model_weights(self) -> None:
        if self.fail_weights:
            raise ModelUnavailableError("Missing weights")

    def analyze_frame(self, _frame, faces=None):
        return {
            "status": "success",
            "face_predictions": [
                {"bbox": [10, 10, 60, 60], "score": 0.2, "is_deepfake": False}
            ],
        }


class _StubTemporalAnalyzer:
    def analyze(self, _tracks):
        return {"status": "ok", "temporal_score": 0.15}


def _make_synthetic_video(video_path: Path, frames: int = 4) -> Path:
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


def test_deepfake_pipeline_sample_frames_and_edge_statuses(tmp_path: Path) -> None:
    frame_a = np.full((128, 128, 3), 200, dtype=np.uint8)
    frame_b = np.full((128, 128, 3), 210, dtype=np.uint8)

    no_model_pipe = DeepfakePipeline(
        audio_analyzer=_StubAudio(),
        video_validator=lambda _p: True,
    )
    sampled = no_model_pipe._sample_frames(
        tmp_path / "stub.mp4",
        [frame_a, {"frame": frame_b, "timestamp": 0.5}],
    )
    assert len(sampled) == 2
    assert sampled[0]["frame_index"] == 0
    assert sampled[1]["timestamp"] == 0.5

    no_model_res = no_model_pipe.run(tmp_path / "stub.mp4", frames=[{"frame": frame_a}])
    assert no_model_res["status"] == "model_unavailable"

    unverified_pipe = DeepfakePipeline(
        visual_detector=_StubVisualDetector(fail_weights=True),
        audio_analyzer=_StubAudio(),
        video_validator=lambda _p: True,
    )
    assert unverified_pipe.run(tmp_path / "stub.mp4", frames=[{"frame": frame_a}])["status"] == "model_unavailable"

    ready_pipe = DeepfakePipeline(
        face_detector=_StubFaceDetectorNoFace(),
        visual_detector=_StubVisualDetector(fail_weights=False),
        audio_analyzer=_StubAudio(),
        video_validator=lambda _p: True,
    )
    assert ready_pipe.run(tmp_path / "stub.mp4", frames=[])["status"] == "no_frames"
    assert ready_pipe.run(tmp_path / "stub.mp4", frames=[{"frame": frame_a}])["status"] == "no_face"


def test_deepfake_pipeline_full_run_and_wrapper(tmp_path: Path) -> None:
    frame_a = np.full((128, 128, 3), 200, dtype=np.uint8)
    res = analyze_video_deepfake(
        video_path=tmp_path / "stub.mp4",
        frames=[{"frame": frame_a, "frame_index": 0, "timestamp": 0.0}],
        face_detector=_StubFaceDetectorWithFace(),
        visual_detector=_StubVisualDetector(fail_weights=False),
        audio_analyzer=_StubAudio(),
        temporal_analyzer=_StubTemporalAnalyzer(),
        risk_scorer=lambda _signals: {"risk_score": 0.2, "decision": "authentic", "is_suspicious": False},
        video_validator=lambda _p: True,
    )
    assert res["status"] == "success"
    assert res["is_deepfake"] is False
    assert res["faces_detected"] == 1
    assert len(res["face_predictions"]) == 1
