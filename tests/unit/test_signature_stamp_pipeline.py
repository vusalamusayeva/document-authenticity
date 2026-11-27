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
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from doc_auth.signature_stamp import pipeline as ss_pipe
from doc_auth.signature_stamp.pipeline import _extract_risk_value, _sanitize_file_id, analyze_signature_stamp


class _DummyBox:
    def __init__(self, coords: list[int]) -> None:
        self._coords = coords

    def as_int_list(self) -> list[int]:
        return list(self._coords)

    def as_list(self) -> list[int]:
        return list(self._coords)


def test_signature_stamp_helpers_and_similarity_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _sanitize_file_id(12345) == "12345"  # type: ignore[arg-type]
    assert _extract_risk_value({"other": 0.9}, ("score", "forgery_probability")) == 0.0

    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    ref = np.full((60, 60, 3), 255, dtype=np.uint8)

    class _FakeDetector:
        def detect_stamps_and_signatures(self, _img: np.ndarray) -> dict[str, Any]:
            return {
                "signatures": [{"bbox": _DummyBox([10, 10, 80, 50]), "confidence": 0.9}],
                "stamps": [],
            }

    scores_iter = iter([0.85, 0.55])

    class _FakeSim:
        @staticmethod
        def compare(_a: Any, _b: Any) -> dict[str, Any]:
            val = next(scores_iter)
            return {"total_similarity": val, "score": val}

    monkeypatch.setattr(ss_pipe, "SignatureStampDetector", _FakeDetector)
    monkeypatch.setattr(ss_pipe, "preprocess_signature", lambda arr, bbox=None: arr)
    monkeypatch.setattr(ss_pipe, "analyze_texture", lambda _crop: {"score": 0.0})
    monkeypatch.setattr(ss_pipe, "SignatureSimilarity", _FakeSim)
    monkeypatch.setattr(ss_pipe, "visualize_results", lambda _img, _cands, out_p: out_p)

    res_match = analyze_signature_stamp(
        img,
        file_id=101,  # type: ignore[arg-type]
        reference_signatures=[ref],
        output_dir=str(tmp_path / "out1"),
    )
    assert res_match.status == "success"
    assert res_match.signature_verifications[0]["status"] == "match"

    res_uncertain = analyze_signature_stamp(
        img,
        file_id=102,  # type: ignore[arg-type]
        reference_signatures=[ref],
        output_dir=str(tmp_path / "out2"),
    )
    assert res_uncertain.status == "success"
    assert res_uncertain.signature_verifications[0]["status"] == "uncertain"
