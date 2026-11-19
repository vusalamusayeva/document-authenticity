from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_auth.deepfake.audio_analysis import AudioLipSyncAnalyzer
from doc_auth.deepfake.exceptions import InvalidConfigError, VideoLoadError
from doc_auth.deepfake.schemas import AudioAnalysisSchema, DeepfakeAnalysisResponse, FacePredictionSchema
from doc_auth.deepfake.video_loader import VideoLoaderConfig, VideoStreamReader, probe_video_metadata, validate_video
from doc_auth.schemas.pipeline_schemas import ArtifactCheckResult, DocumentAuthSummary


def test_schemas_instantiation() -> None:
    fp = FacePredictionSchema(face_index=0, bbox=[0, 0, 50, 50], score=0.75, is_deepfake=True)
    assert fp.score == 0.75

    aud = AudioAnalysisSchema(status="no_audio", has_audio=False)
    assert aud.has_audio is False

    resp = DeepfakeAnalysisResponse(status="success", is_deepfake=False, risk_score=0.2)
    assert resp.status == "success"

    chk = ArtifactCheckResult(module="visual", status="success", score=0.3)
    summary = DocumentAuthSummary(status="success", overall_risk_score=0.3, checks=[chk])
    assert len(summary.checks) == 1


def test_video_loader_synthetic_avi(tmp_path: Path) -> None:
    video_file = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_file), fourcc, 10.0, (160, 120))
    for i in range(5):
        frame = np.full((120, 160, 3), i * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    if not video_file.exists() or video_file.stat().st_size == 0:
        pytest.skip("OpenCV MJPG writer bu mühitdə fayl yaratmadı.")

    meta = validate_video(video_file, VideoLoaderConfig(max_width=320, max_height=240))
    assert meta["width"] == 160
    assert meta["height"] == 120

    with VideoStreamReader(video_file) as reader:
        first = reader.read_frame()
        assert first is not None
        assert first.shape[:2] == (120, 160)


def test_video_loader_invalid_cases(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(VideoLoadError):
        probe_video_metadata(empty_file)

    with pytest.raises(InvalidConfigError):
        validate_video("any.mp4", VideoLoaderConfig(probe_timeout_sec=0.0))


def test_audio_analyzer_init_validation() -> None:
    with pytest.raises(ValueError):
        AudioLipSyncAnalyzer(sample_rate=0)