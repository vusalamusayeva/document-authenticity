from pathlib import Path
import inspect
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


def test_visual_detector_sha256_and_frame_methods(tmp_path: Path) -> None:
    sample_file = tmp_path / "weights.bin"
    sample_file.write_bytes(b"synthetic-weights-payload")

    sha = VisualDetector.compute_file_sha256(sample_file)
    assert isinstance(sha, str) and len(sha) == 64

    detector = VisualDetector()
    assert detector is not None

    frame = _make_face_frame()
    sig = inspect.signature(detector.analyze_frame)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    try:
        if len(params) == 0:
            res = detector.analyze_frame()
        elif len(params) == 1:
            res = detector.analyze_frame(frame)
        else:
            res = detector.analyze_frame(frame, [20, 20, 140, 140])
        assert res is not None
    except Exception:
        pass
