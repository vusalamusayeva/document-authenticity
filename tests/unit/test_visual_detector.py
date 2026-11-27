from pathlib import Path
import cv2
import numpy as np

from doc_auth.deepfake.visual_detector import VisualDetector


def _make_face_frame() -> np.ndarray:
    frame = np.full((160, 160, 3), 210, dtype=np.uint8)
    cv2.circle(frame, (80, 80), 42, (165, 130, 110), -1)
    cv2.circle(frame, (65, 70), 5, (30, 30, 30), -1)
    cv2.circle(frame, (95, 70), 5, (30, 30, 30), -1)
    cv2.ellipse(frame, (80, 98), (14, 7), 0, 0, 180, (40, 40, 120), 2)
    return frame


def test_visual_detector_sha256_and_edge_frames(tmp_path: Path) -> None:
    sample_file = tmp_path / "weights.bin"
    sample_file.write_bytes(b"synthetic-weights-payload")

    sha = VisualDetector.compute_file_sha256(sample_file)
    assert isinstance(sha, str) and len(sha) == 64

    detector = VisualDetector()
    invalid_res = detector.analyze_frame(None)  # type: ignore[arg-type]
    assert invalid_res["status"] == "invalid_frame"

    frame = _make_face_frame()
    no_face_res = detector.analyze_frame(frame, faces=[])
    assert no_face_res["status"] == "no_face"

    empty_patch = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector._compute_region_score(empty_patch) == 0.0


def test_visual_detector_features_scoring_and_analyze_frame() -> None:
    detector = VisualDetector()
    frame = _make_face_frame()

    feats = detector._extract_features(frame)
    assert feats.shape == (4,)
    assert feats.dtype == np.float32

    detector._weights_data = {
        "weights": np.array([0.1, -0.1, 0.05, 0.2], dtype=np.float32),
        "bias": np.array([-0.5], dtype=np.float32),
        "laplacian_scale": np.array([1.0], dtype=np.float32),
    }

    score = detector._compute_region_score(frame)
    assert 0.0 <= score <= 1.0

    res = detector.analyze_frame(frame, faces=[{"bbox": [25, 25, 135, 135]}])
    assert res["status"] == "success"
    assert res["face_detected"] is True
    assert len(res["face_predictions"]) == 1
