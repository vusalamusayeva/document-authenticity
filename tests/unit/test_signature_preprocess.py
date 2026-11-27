import cv2
import numpy as np
import pytest

from doc_auth.signature_stamp.exceptions import SignatureStampError
from doc_auth.signature_stamp.signature_preprocess import preprocess_signature


def test_preprocess_signature_branches_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SignatureStampError):
        preprocess_signature(np.array([], dtype=np.uint8))

    # Empty crop fallback (ymin >= ymax)
    img = np.full((100, 140, 3), 240, dtype=np.uint8)
    cv2.line(img, (20, 50), (110, 50), (10, 10, 10), 4)
    res_empty_bbox = preprocess_signature(img, bbox=(80, 10, 40, 60))
    assert res_empty_bbox.shape == (150, 300)

    # Valid bbox crop + grayscale/binary ROI normalization
    res_valid = preprocess_signature(img, bbox=(10, 10, 90, 130))
    assert res_valid.shape == (150, 300)
    assert res_valid.dtype == np.uint8

    # Zero-dimension ROI branch when resized ROI has 0 height/width
    monkeypatch.setattr(cv2, "resize", lambda *args, **kwargs: np.zeros((0, 0), dtype=np.uint8))
    res_zero_roi = preprocess_signature(img)
    assert res_zero_roi.shape == (150, 300)
    assert np.all(res_zero_roi == 255)

    # Internal exception wrapped in SignatureStampError
    monkeypatch.setattr(
        cv2,
        "threshold",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("thresh error")),
    )
    with pytest.raises(SignatureStampError):
        preprocess_signature(img)
