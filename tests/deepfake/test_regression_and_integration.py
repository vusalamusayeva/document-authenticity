import io
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from doc_auth.deepfake.aggregation import aggregate_visual_predictions
from doc_auth.deepfake.api import attach_deepfake_router
from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer
from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError
from doc_auth.deepfake.face_detector import FaceDetector
from doc_auth.deepfake.frame_sampler import FrameSamplingConfig
from doc_auth.deepfake.pipeline import DeepfakePipeline
from doc_auth.deepfake.risk_scoring import RiskScorer, calculate_risk_score, validate_score_value
from doc_auth.deepfake.temporal_analysis import TemporalAnalyzer, assign_face_tracks
from doc_auth.deepfake.visual_detector import VisualDetector
from doc_auth.service import AnalysisService


def test_missing_or_empty_checkpoint_raises_model_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / "weights_missing.pth"
    det_missing = VisualDetector(checkpoint_path=missing)
    with pytest.raises(ModelUnavailableError):
        det_missing.verify_model_weights()

    empty = tmp_path / "weights_empty.onnx"
    empty.write_bytes(b"")
    det_empty = VisualDetector(checkpoint_path=empty)
    with pytest.raises(ModelUnavailableError):
        det_empty.verify_model_weights()


def test_no_face_and_no_frames_never_return_fake_or_real(tmp_path: Path) -> None:
    ckpt = tmp_path / "dummy.pth"
    ckpt.write_bytes(b"fake-bytes-for-path-test")

    class StubDetector(VisualDetector):
        def verify_model_weights(self) -> None:
            return None

        def load_model(self):
            return object()

    class EmptyFaceDetector(FaceDetector):
        def detect_faces(self, frame: np.ndarray):
            return []

    class SilentAudio(AudioLipSyncAnalyzer):
        def analyze(self, video_path):
            return {
                "status": "no_audio",
                "has_audio": False,
                "audio_duration": None,
                "video_duration": 2.0,
                "duration_diff": None,
                "lip_sync_score": None,
                "is_suspicious": None,
            }

    # 1) Üzsüz frame -> status="no_face", is_deepfake=None (False / "real" deyil!)
    pipe_no_face = DeepfakePipeline(
        face_detector=EmptyFaceDetector(),
        visual_detector=StubDetector(checkpoint_path=ckpt),
        audio_analyzer=SilentAudio(),
    )
    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    res_no_face = pipe_no_face.run(tmp_path / "video.mp4", frames=[{"frame": blank, "timestamp": 0.0}])
    assert res_no_face["status"] == "no_face"
    assert res_no_face["is_deepfake"] is None
    assert res_no_face["faces_detected"] == 0
    assert res_no_face["audio"]["has_audio"] is False

    # 2) Boş sampling -> status="no_frames", is_deepfake=None
    pipe_no_frames = DeepfakePipeline(
        face_detector=EmptyFaceDetector(),
        visual_detector=StubDetector(checkpoint_path=ckpt),
        audio_analyzer=SilentAudio(),
        frame_sampler=lambda _p: [],
    )
    res_no_frames = pipe_no_frames.run(tmp_path / "video.mp4")
    assert res_no_frames["status"] == "no_frames"
    assert res_no_frames["is_deepfake"] is None
    assert res_no_frames["frames_analyzed"] == 0


def test_multi_face_same_frame_and_scene_cut_in_temporal_analysis() -> None:
    # Eyni frame-də iki fərqli üz ayrı track_id alır və ardıcıl zaman addımı kimi müqayisə edilmir
    frame_items = [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "faces": [
                {"bbox": [10, 10, 60, 60], "score": 0.10},
                {"bbox": [150, 10, 200, 60], "score": 0.90},
            ],
        },
        {
            "frame_index": 1,
            "timestamp": 0.2,
            "faces": [
                {"bbox": [12, 12, 62, 62], "score": 0.12},
                {"bbox": [152, 12, 202, 62], "score": 0.88},
            ],
        },
        # Səhnə dəyişməsi / zaman boşluğu (dt = 5.0s > max_time_gap=1.0s)
        {
            "frame_index": 2,
            "timestamp": 5.2,
            "faces": [
                {"bbox": [12, 12, 62, 62], "score": 0.95},
            ],
        },
    ]

    tracked = assign_face_tracks(frame_items, iou_threshold=0.3, max_time_gap=1.0)
    track_ids_f0 = {x["track_id"] for x in tracked if x["frame_index"] == 0}
    assert len(track_ids_f0) == 2

    analyzer = TemporalAnalyzer(max_time_gap=1.0, high_jump_threshold=0.35)
    temp_res = analyzer.analyze(tracked)
    assert temp_res["status"] == "success"
    assert temp_res["scene_cuts_ignored"] >= 1
    # Hər track daxilində skor sıçrayışı çox kiçikdir (0.10->0.12 və 0.90->0.88)
    assert temp_res["temporal_risk"] < 0.25

    agg = aggregate_visual_predictions(tracked, frames_count=3)
    assert agg["frames_count"] == 3
    assert agg["faces_count"] == 5
    assert agg["max_score"] == pytest.approx(0.95)


def test_invalid_configs_and_invalid_scores_raise_errors() -> None:
    with pytest.raises(InvalidConfigError):
        FrameSamplingConfig(sampling_fps=0.0, max_frames=10)
    with pytest.raises(InvalidConfigError):
        FrameSamplingConfig(sampling_fps=2.0, max_frames=-5)
    with pytest.raises(InvalidConfigError):
        RiskScorer(visual_weight=-0.2, temporal_weight=0.5, audio_weight=0.5)

    with pytest.raises(ValueError):
        validate_score_value(1.5, "score")
    with pytest.raises(ValueError):
        validate_score_value(-0.1, "score")
    with pytest.raises(ValueError):
        validate_score_value(float("nan"), "score")

    with pytest.raises(ValueError):
        calculate_risk_score({
            "visual": {"status": "success", "score": 1.2},
        })


def test_api_upload_size_limit_and_decision_flow() -> None:
    app = FastAPI()
    attach_deepfake_router(
        app,
        pipeline_runner=lambda _path: {"status": "success", "is_deepfake": True, "risk_score": 0.85},
        max_upload_bytes=64,
    )
    client = TestClient(app)

    # Limitdən böyük fayl -> 413
    oversized = io.BytesIO(b"x" * 128)
    resp_large = client.post(
        "/deepfake/analyze",
        files={"file": ("clip.mp4", oversized, "video/mp4")},
    )
    assert resp_large.status_code == 413

    # Limit daxilində fayl -> 200 və is_deepfake=True
    valid_payload = io.BytesIO(b"valid-test-bytes")
    resp_ok = client.post(
        "/deepfake/analyze",
        files={"file": ("clip.mp4", valid_payload, "video/mp4")},
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["is_deepfake"] is True


def test_analysis_service_end_to_end_real_flow(tmp_path: Path) -> None:
    doc = tmp_path / "sample_doc.txt"
    doc.write_text("Official document content for verification.", encoding="utf-8")

    class StubComponent:
        def __init__(self, out):
            self.out = out

        def analyze(self, _inp):
            return self.out

    service = AnalysisService(
        loader=StubComponent({"status": "success", "pages": 1}),
        detector=StubComponent({"status": "success", "detected": True}),
        metadata_analyzer=StubComponent({"status": "success", "author": "system"}),
        ocr_checker=StubComponent({"status": "success", "text": "Official document"}),
        classifier=StubComponent({"status": "success", "doc_type": "report"}),
        tampering_detector=StubComponent({"status": "success", "tampered": False}),
        decision_model=StubComponent({"status": "success", "authentic": True}),
        explainer=StubComponent({"status": "success", "summary": "All checks passed."}),
    )

    result = service.analyze(doc)
    assert result["status"] == "success"
    assert result["decision"]["authentic"] is True
    assert result["explanation"]["summary"] == "All checks passed."