from pathlib import Path
from typing import Any

import numpy as np
import pytest

from doc_auth.deepfake.pipeline import DeepfakePipeline, analyze_video_deepfake


class _StubFaceDetector:
    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return [{"bbox": [1, 1, 10, 10]}]


class _StubVisualDetector:
    def __init__(self, threshold: float = 0.5, score: float = 0.8) -> None:
        self.threshold = threshold
        self._score = score

    def verify_model_weights(self) -> bool:
        return True

    def analyze_frame(self, frame: np.ndarray, faces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "status": "success",
            "face_predictions": [
                {"bbox": [1, 1, 10, 10], "score": self._score, "is_deepfake": self._score >= self.threshold}
            ],
        }


class _FailingAudio:
    def analyze(self, path: Path) -> dict[str, Any]:
        raise RuntimeError("audio boom")


class _FailingTemporal:
    def analyze(self, tracks: Any) -> dict[str, Any]:
        raise RuntimeError("temporal boom")


def _str_only_validator(p: str) -> None:
    if not isinstance(p, str):
        raise TypeError("expects str")


def _str_only_sampler(p: str) -> list[dict[str, Any]]:
    if not isinstance(p, str):
        raise TypeError("expects str")
    return [{"frame": np.zeros((12, 12, 3), dtype=np.uint8)}]


def _failing_aggregator(preds: list[dict[str, Any]]) -> dict[str, Any]:
    raise RuntimeError("agg boom")


def test_pipeline_typeerror_fallbacks_and_module_error_paths(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")

    pipe = DeepfakePipeline(
        face_detector=_StubFaceDetector(),  # type: ignore[arg-type]
        visual_detector=_StubVisualDetector(threshold=0.5, score=0.75),  # type: ignore[arg-type]
        audio_analyzer=_FailingAudio(),  # type: ignore[arg-type]
        frame_sampler=_str_only_sampler,
        temporal_analyzer=_FailingTemporal(),  # type: ignore[arg-type]
        risk_scorer=lambda signals: {"risk_score": None, "decision": "inconclusive"},
        video_validator=_str_only_validator,
        aggregator=_failing_aggregator,
    )

    res = pipe.run(video)
    assert res["status"] == "partial"
    assert res["is_deepfake"] is None
    assert res["module_statuses"]["audio"] == "error"
    assert res["module_statuses"]["temporal"] == "error"
    assert res["module_statuses"]["aggregation"] == "fallback_max"
    assert res["visual_score"] == pytest.approx(0.75)


def test_pipeline_without_risk_scorer_and_validator_exception(tmp_path: Path) -> None:
    video = tmp_path / "clip2.mp4"
    video.write_bytes(b"fake-video")

    # Validator raising non-VideoValidationError exception
    pipe_val_err = DeepfakePipeline(
        video_validator=lambda p: (_ for _ in ()).throw(ValueError("bad meta")),
    )
    err_res = pipe_val_err.run(video)
    assert err_res["status"] == "failed"
    assert err_res["error_type"] == "ValueError"

    # risk_scorer=None uses visual_agg_score and visual_detector.threshold
    pipe_no_scorer = DeepfakePipeline(
        face_detector=_StubFaceDetector(),  # type: ignore[arg-type]
        visual_detector=_StubVisualDetector(threshold=0.6, score=0.85),  # type: ignore[arg-type]
        audio_analyzer=_FailingAudio(),  # type: ignore[arg-type]
        frame_sampler=lambda p: [np.ones((10, 10, 3), dtype=np.uint8)],
        risk_scorer=None,  # type: ignore[arg-type]
        video_validator=lambda p: None,
        aggregator=lambda preds: {"aggregated_score": 0.85},
    )
    out = analyze_video_deepfake(
        video_path=video,
        face_detector=_StubFaceDetector(),  # type: ignore[arg-type]
        visual_detector=_StubVisualDetector(threshold=0.6, score=0.85),  # type: ignore[arg-type]
        audio_analyzer=_FailingAudio(),  # type: ignore[arg-type]
        frame_sampler=lambda p: [np.ones((10, 10, 3), dtype=np.uint8)],
        risk_scorer=None,  # type: ignore[arg-type]
        video_validator=lambda p: None,
        aggregator=lambda preds: {"aggregated_score": 0.85},
    )
    assert out["status"] == "success"
    assert out["is_deepfake"] is True
    assert out["risk_score"] == pytest.approx(0.85)
    assert pipe_no_scorer is not None
