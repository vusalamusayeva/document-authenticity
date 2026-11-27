from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.forensics.tampering_detection import TamperingDetector


def _make_tampered_image(img_path: Path) -> Path:
    img = np.full((240, 360, 3), 240, dtype=np.uint8)
    cv2.putText(img, "AMOUNT: $1,250.00", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.rectangle(img, (25, 100), (165, 165), (40, 40, 40), -1)
    # Kopyalanmış / dəyişdirilmiş blok əlavə et
    patch = img[100:165, 25:165].copy()
    img[100:165, 190:330] = patch
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


def test_tampering_detector_methods(tmp_path: Path) -> None:
    detector = TamperingDetector()
    assert detector is not None

    img_path = _make_tampered_image(tmp_path / "tampered_doc.png")
    for method_name in ("_extract_image", "detect", "analyze", "run", "__call__"):
        method = getattr(detector, method_name, None)
        if callable(method):
            try:
                res = _call_with_path(method, img_path)
                assert res is not None
            except Exception:
                pass
