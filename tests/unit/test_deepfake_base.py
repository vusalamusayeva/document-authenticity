import numpy as np
import pytest

from doc_auth.deepfake.base import BaseDeepfakeDetector


class _DummyDeepfakeDetector(BaseDeepfakeDetector):
    def analyze_frame(self, frame: np.ndarray):
        return {
            "shape": frame.shape,
            "is_fake": False,
            "confidence": 0.12,
        }


def test_base_deepfake_detector_abstract_and_concrete() -> None:
    with pytest.raises(TypeError):
        BaseDeepfakeDetector()  # type: ignore[abstract]

    detector = _DummyDeepfakeDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = detector.analyze_frame(frame)
    assert result["shape"] == (64, 64, 3)
    assert result["is_fake"] is False
    assert 0.0 <= result["confidence"] <= 1.0
