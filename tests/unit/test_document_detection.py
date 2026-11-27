from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_auth.ingestion.document_detection import DocumentDetector


def test_init_resolve_and_passthrough(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DocumentDetector(min_area_ratio=0.0)
    with pytest.raises(ValueError):
        DocumentDetector(min_area_ratio=1.0)

    det = DocumentDetector(min_area_ratio=0.05)
    assert det._resolve_image(np.array([], dtype=np.uint8)) is None
    assert det._resolve_image(tmp_path / "missing.png") is None

    pass_res = det.detect({"custom": 123})
    assert pass_res["status"] == "passthrough"
    assert pass_res["detected"] is None
    assert pass_res["custom"] == 123

    img = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (140, 100), (10, 10, 10), 3)
    img_path = tmp_path / "doc_sample.png"
    assert cv2.imwrite(str(img_path), img)

    loaded = det._resolve_image({"path": str(img_path)})
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (120, 160, 3)


def test_detect_contour_and_method_aliases() -> None:
    det = DocumentDetector(min_area_ratio=0.05)
    canvas = np.full((160, 200, 3), 250, dtype=np.uint8)
    cv2.rectangle(canvas, (25, 25), (175, 135), (0, 0, 0), 3)

    res = det.detect({"image": canvas, "file_name": "doc.png", "extension": ".png"})
    assert res["status"] == "success"
    assert res["detected"] is True
    assert res["has_contour"] is True
    assert res["file_name"] == "doc.png"
    assert res["extension"] == ".png"
    assert len(res["bbox"]) == 4

    assert det.detect_document(canvas)["status"] == "success"
    assert det.analyze(canvas)["status"] == "success"
    assert det(canvas)["status"] == "success"

    # High threshold with blank image -> no contour above threshold
    det_strict = DocumentDetector(min_area_ratio=0.9)
    blank_gray = np.full((64, 64), 128, dtype=np.uint8)
    res_blank = det_strict.detect(blank_gray)
    assert res_blank["has_contour"] is False
    assert res_blank["bbox"] == [0, 0, 64, 64]
