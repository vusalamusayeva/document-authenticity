import cv2
import numpy as np
import pytest

from doc_auth.signature_stamp.exceptions import StampAnalysisError
from doc_auth.signature_stamp.schemas import StampResult
from doc_auth.signature_stamp.stamp_analysis import analyze_stamp


def test_analyze_stamp_edge_and_error_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_res = analyze_stamp(np.array([], dtype=np.uint8))
    assert isinstance(empty_res, StampResult)
    assert empty_res.detected is False

    # Flat white image -> no dark foreground contour
    white_img = np.full((80, 80, 3), 255, dtype=np.uint8)
    no_contour = analyze_stamp(white_img)
    assert no_contour.detected is False

    # Tiny dark dot -> area < 200
    tiny_img = np.full((80, 80, 3), 255, dtype=np.uint8)
    cv2.circle(tiny_img, (40, 40), 4, (0, 0, 0), -1)
    tiny_res = analyze_stamp(tiny_img)
    assert tiny_res.detected is False
    assert tiny_res.confidence == pytest.approx(0.1)

    # Internal exception wrapped in StampAnalysisError
    monkeypatch.setattr(
        cv2,
        "threshold",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("thresh fail")),
    )
    with pytest.raises(StampAnalysisError):
        analyze_stamp(white_img)


def test_analyze_stamp_valid_stamp() -> None:
    stamp_img = np.full((160, 160, 3), 245, dtype=np.uint8)
    cv2.circle(stamp_img, (80, 80), 48, (200, 30, 10), -1)

    res = analyze_stamp(stamp_img)
    assert isinstance(res, StampResult)
    assert res.detected is True
    assert 0.0 <= res.confidence <= 1.0
