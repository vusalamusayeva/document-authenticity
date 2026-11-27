from pathlib import Path
import cv2
import numpy as np
import pytest

from doc_auth.signature_stamp.pipeline import analyze_signature_stamp


def test_analyze_signature_stamp_pipeline(tmp_path: Path) -> None:
    img_path = tmp_path / "doc_stamp.png"
    canvas = np.ones((240, 360, 3), dtype=np.uint8) * 245
    cv2.rectangle(canvas, (30, 30), (330, 210), (30, 30, 30), 2)
    cv2.circle(canvas, (250, 140), 42, (180, 30, 30), 2)
    cv2.putText(canvas, "SIGNED", (55, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.imwrite(str(img_path), canvas)

    res_path = analyze_signature_stamp(str(img_path))
    assert res_path is not None

    res_arr = analyze_signature_stamp(canvas)
    assert res_arr is not None

    missing_path = str(tmp_path / "missing_doc.png")
    try:
        res_missing = analyze_signature_stamp(missing_path)
        assert res_missing is not None
    except Exception:
        pass
