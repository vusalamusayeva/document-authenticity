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
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from doc_auth.deepfake.exceptions import ModelUnavailableError, VideoValidationError
from doc_auth.deepfake.pipeline import DeepfakePipeline


class _DummyFaceDetector:
    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return [{"bbox": [2, 2, 20, 20], "score": 0.9}]


class _DummyVisualDetector:
    def __init__(self, threshold: float = 0.5, fail_verify: bool = False) -> None:
        self.threshold = threshold
        self._fail_verify = fail_verify

    def verify_model_weights(self) -> bool:
        if self._fail_verify:
            raise ModelUnavailableError("weights missing")
        return True

    def analyze_frame(self, frame: np.ndarray, faces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "status": "success",
            "face_predictions": [{"face_index": 0, "bbox": [2, 2, 20, 20], "score": 0.85, "is_deepfake": True}],
        }


class _DummyAudioAnalyzer:
    def analyze(self, path: Path) -> dict[str, Any]:
        return {"status": "ok", "score": 0.1}


class _DummyTemporalAnalyzer:
    def analyze(self, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        return {"status": "ok", "temporal_score": 0.2}


def test_pipeline_remaining_branches(tmp_path: Path) -> None:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake-video")

    # 1. video_validator=None + risk_scorer=None
    p_no_val_no_scorer = DeepfakePipeline(
        face_detector=_DummyFaceDetector(),  # type: ignore[arg-type]
        visual_detector=_DummyVisualDetector(threshold=0.5),  # type: ignore[arg-type]
        audio_analyzer=_DummyAudioAnalyzer(),  # type: ignore[arg-type]
        frame_sampler=lambda _p: [{"frame": np.zeros((24, 24, 3), dtype=np.uint8), "frame_index": 0, "timestamp": 0.0}],
        temporal_analyzer=_DummyTemporalAnalyzer(),  # type: ignore[arg-type]
        risk_scorer=None,
        video_validator=lambda _p: None,
    )
    p_no_val_no_scorer.video_validator = None
    p_no_val_no_scorer.risk_scorer = None
    out_direct = p_no_val_no_scorer.run(vid)
    assert out_direct["status"] == "success"
    assert out_direct["risk_score"] == 0.85
    assert out_direct["is_deepfake"] is True

    # 2. VideoValidationError during validation
    def _raise_val_err(_p: Any) -> None:
        raise VideoValidationError("invalid container")

    p_val_err = DeepfakePipeline(video_validator=_raise_val_err)
    out_val_err = p_val_err.run(vid)
    assert out_val_err["status"] == "failed"
    assert out_val_err["error_type"] == "VideoValidationError"

    # 3. visual_detector is None
    p_no_vis = DeepfakePipeline(
        audio_analyzer=_DummyAudioAnalyzer(),  # type: ignore[arg-type]
        video_validator=lambda _p: None,
    )
    out_no_vis = p_no_vis.run(vid)
    assert out_no_vis["status"] == "model_unavailable"

    # 4. verify_model_weights raises ModelUnavailableError
    p_bad_weights = DeepfakePipeline(
        visual_detector=_DummyVisualDetector(fail_verify=True),  # type: ignore[arg-type]
        audio_analyzer=_DummyAudioAnalyzer(),  # type: ignore[arg-type]
        video_validator=lambda _p: None,
    )
    out_bad_weights = p_bad_weights.run(vid)
    assert out_bad_weights["status"] == "model_unavailable"
    assert out_bad_weights["error_type"] == "ModelUnavailableError"

    # 5. VideoValidationError during frame sampling
    def _sampler_err(_p: Any) -> Any:
        raise VideoValidationError("sampler failed")

    p_samp_err = DeepfakePipeline(
        visual_detector=_DummyVisualDetector(),  # type: ignore[arg-type]
        audio_analyzer=_DummyAudioAnalyzer(),  # type: ignore[arg-type]
        frame_sampler=_sampler_err,
        video_validator=lambda _p: None,
    )
    out_samp_err = p_samp_err.run(vid)
    assert out_samp_err["status"] == "failed"
    assert out_samp_err["module_statuses"]["sampler"] == "failed"
