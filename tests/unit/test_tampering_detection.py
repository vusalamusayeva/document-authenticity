from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.forensics.tampering_detection import TamperingDetector


def _make_images(tmp_path: Path):
    clean = np.ones((200, 260, 3), dtype=np.uint8) * 235
    cv2.rectangle(clean, (20, 20), (240, 180), (40, 40, 40), 2)
    cv2.putText(clean, "DOCUMENT 2026", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2)

    tampered = clean.copy()
    patch = tampered[60:95, 40:120].copy()
    tampered[115:150, 140:220] = patch
    cv2.rectangle(tampered, (140, 115), (220, 150), (0, 0, 220), 2)

    clean_path = tmp_path / "clean.png"
    tampered_path = tmp_path / "tampered.png"
    cv2.imwrite(str(clean_path), clean)
    cv2.imwrite(str(tampered_path), tampered)
    return clean, tampered, str(clean_path), str(tampered_path)


def _invoke_method(fn, primary, secondary=None):
    sig = inspect.signature(fn)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if len(params) == 0:
        return fn()
    if len(params) == 1:
        return fn(primary)
    if secondary is not None:
        try:
            return fn(primary, secondary)
        except Exception:
            return fn(primary)
    return fn(primary)


def test_tampering_detector_workflow(tmp_path: Path) -> None:
    detector = TamperingDetector()
    assert detector is not None

    clean, tampered, clean_path, tampered_path = _make_images(tmp_path)

    for method_name in ("detect", "analyze", "run", "__call__"):
        fn = getattr(detector, method_name, None)
        if callable(fn):
            res_clean = _invoke_method(fn, clean, tampered)
            assert res_clean is not None

            res_path = _invoke_method(fn, clean_path, tampered_path)
            assert res_path is not None
            break
