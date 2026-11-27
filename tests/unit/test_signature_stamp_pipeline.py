from pathlib import Path
import cv2
import numpy as np
import pytest

from doc_auth.signature_stamp.exceptions import SignatureStampError
from doc_auth.signature_stamp.pipeline import (
    _extract_risk_value,
    _sanitize_file_id,
    analyze_signature_stamp,
)


def _make_doc_image() -> np.ndarray:
    img = np.ones((260, 380, 3), dtype=np.uint8) * 245
    cv2.putText(img, "APPROVED DOCUMENT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2)
    cv2.ellipse(img, (270, 160), (45, 45), 0, 0, 360, (180, 30, 30), 2)
    cv2.putText(img, "SEAL", (248, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 30, 30), 2)
    cv2.line(img, (40, 175), (155, 140), (30, 30, 160), 2)
    cv2.line(img, (155, 140), (195, 170), (30, 30, 160), 2)
    return img


def test_sanitize_and_extract_helpers() -> None:
    assert _sanitize_file_id("  doc_123  ") == "doc_123"
    with pytest.raises(SignatureStampError):
        _sanitize_file_id("../secret")
    with pytest.raises(SignatureStampError):
        _sanitize_file_id("")

    assert _extract_risk_value({"score": 1.5}, ("score",)) == 1.0
    assert _extract_risk_value({"score": -0.2}, ("score",)) == 0.0
    assert _extract_risk_value(None, ("score",)) == 0.0


def test_analyze_signature_stamp_workflow(tmp_path: Path) -> None:
    img = _make_doc_image()
    out_dir = tmp_path / "outputs"

    result = analyze_signature_stamp(
        image=img,
        file_id="doc_001",
        reference_signatures=[img[130:185, 35:205].copy()],
        output_dir=str(out_dir),
    )

    assert result is not None
