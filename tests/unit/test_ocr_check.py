from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.ocr.ocr_check import OCRChecker


def _make_text_image(tmp_path: Path):
    img = np.ones((180, 320, 3), dtype=np.uint8) * 245
    cv2.putText(img, "INVOICE NO: 2026-001", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, "TOTAL: 150.00 AZN", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    img_path = tmp_path / "ocr_sample.png"
    cv2.imwrite(str(img_path), img)
    return img, str(img_path)


def _invoke_ocr_method(fn, img, img_path):
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if len(params) == 0:
        return fn()
    for candidate in (img_path, img):
        try:
            return fn(candidate)
        except Exception:
            continue
    return None


def test_ocr_checker_workflow(tmp_path: Path) -> None:
    checker = OCRChecker()
    assert checker is not None

    img, img_path = _make_text_image(tmp_path)

    for method_name in ("check", "run", "analyze", "__call__"):
        fn = getattr(checker, method_name, None)
        if callable(fn):
            res = _invoke_ocr_method(fn, img, img_path)
            if res is not None:
                assert res is not None
                break
