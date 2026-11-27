from pydantic import ValidationError
import pytest

from doc_auth.signature_stamp.schemas import BoundingBox, SignatureStampResult


def test_bounding_box_dimensions_and_result_file_id_validation() -> None:
    box = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0, confidence=0.9)
    assert box.width == pytest.approx(40.0)
    assert box.height == pytest.approx(60.0)

    clamped = BoundingBox(x1=50.0, y1=80.0, x2=10.0, y2=20.0)
    assert clamped.width == pytest.approx(0.0)
    assert clamped.height == pytest.approx(0.0)

    ok_res = SignatureStampResult(file_id=42)  # type: ignore[arg-type]
    assert ok_res.file_id == "42"

    for bad_id in (None, "", "   "):
        with pytest.raises(ValidationError):
            SignatureStampResult(file_id=bad_id)  # type: ignore[arg-type]
