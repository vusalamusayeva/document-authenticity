import pytest
import numpy as np
from src.doc_auth.deepfake.visual_detector import VisualDetector

def test_visual_detector_frame():
    detector = VisualDetector()
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    res = detector.analyze_frame(dummy_frame)
    assert "face_detected" in res
    assert "max_deepfake_score" in res
    assert 0.0 <= res["max_deepfake_score"] <= 1.0