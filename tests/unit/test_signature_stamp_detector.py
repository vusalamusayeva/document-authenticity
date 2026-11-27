import cv2
import numpy as np
import pytest

from doc_auth.signature_stamp.detector import SignatureStampDetector
from doc_auth.signature_stamp.exceptions import SignatureStampError
from doc_auth.signature_stamp.schemas import BoundingBox


def test_detector_errors_and_empty_signature() -> None:
    det = SignatureStampDetector(confidence_threshold=0.3)

    with pytest.raises(SignatureStampError):
        det.detect_stamps_and_signatures(np.array([], dtype=np.uint8))

    blank = np.full((120, 120, 3), 255, dtype=np.uint8)
    assert det.detect_signature(blank) is None

    # Force internal error inside detection loop
    bad_img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(SignatureStampError):
        det.detect_stamps_and_signatures(bad_img)  # type: ignore[arg-type]


def test_detector_finds_stamp_and_signature() -> None:
    det = SignatureStampDetector(confidence_threshold=0.5)
    canvas = np.full((320, 420, 3), 255, dtype=np.uint8)

    # Filled circle -> circularity close to 1.0 -> detected as stamp
    cv2.circle(canvas, (90, 90), 42, (180, 40, 20), -1)

    # Wide horizontal stroke -> aspect_ratio > 1.2 and area > 800 -> detected as signature
    cv2.rectangle(canvas, (170, 200), (370, 235), (20, 20, 20), -1)

    res = det.detect_stamps_and_signatures(canvas)
    assert len(res["stamps"]) >= 1
    assert len(res["signatures"]) >= 1

    sig_box = det.detect_signature(canvas)
    assert isinstance(sig_box, BoundingBox)
