import cv2
import numpy as np

from doc_auth.signature_stamp.stamp_analysis import analyze_stamp


def test_analyze_stamp_synthetic_circle():
    # Sintetik dairəvi möhür çəkilir (vahid boz şəkil əvəzinə)
    stamp_img = np.ones((150, 150, 3), dtype=np.uint8) * 255
    cv2.circle(stamp_img, (75, 75), 50, (255, 0, 0), 3)
    
    result = analyze_stamp(stamp_img)
    assert result is not None
    assert result.detected is True
    assert result.confidence > 0.4
    assert result.circularity is not None
    assert result.solidity is not None

def test_stamp_analysis_reraises_stamp_error(monkeypatch):
    import pytest
    import doc_auth.signature_stamp.stamp_analysis as sa_mod
    from doc_auth.signature_stamp.exceptions import StampAnalysisError

    img = np.ones((100, 100, 3), dtype=np.uint8) * 255

    def _boom_contours(*args, **kwargs):
        raise StampAnalysisError("domain stamp failure")

    monkeypatch.setattr(sa_mod.cv2, "findContours", _boom_contours)
    with pytest.raises(StampAnalysisError, match="domain stamp failure"):
        sa_mod.analyze_stamp_region(img)
