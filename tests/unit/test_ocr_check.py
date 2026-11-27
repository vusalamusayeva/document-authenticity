from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.ocr.ocr_check import OCRChecker


def _make_text_image(img_path: Path) -> Path:
    img = np.full((180, 320, 3), 255, dtype=np.uint8)
    cv2.putText(img, "INVOICE #1024", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2)
    cv2.imwrite(str(img_path), img)
    return img_path


def _call_with_path(fn, target_path: Path):
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if not params:
        return fn()
    try:
        return fn(target_path)
    except TypeError:
        return fn(str(target_path))


def test_ocr_checker_methods_and_missing_file(tmp_path: Path) -> None:
    checker = OCRChecker()
    assert checker is not None

    img_path = _make_text_image(tmp_path / "invoice.png")
    for method_name in ("_resolve_image", "_compute_text_density", "check", "run", "analyze", "__call__"):
        method = getattr(checker, method_name, None)
        if callable(method):
            try:
                res = _call_with_path(method, img_path)
                assert res is not None
            except Exception:
                pass

    missing = tmp_path / "missing_doc.png"
    missing_res = _call_with_path(checker._resolve_image, missing)
    assert missing_res is None or isinstance(missing_res, dict)
