from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth.signature_stamp.pipeline import analyze_signature_stamp


def _make_doc_image(tmp_path: Path):
    img = np.ones((240, 360, 3), dtype=np.uint8) * 245
    cv2.putText(img, "APPROVED DOCUMENT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2)
    cv2.ellipse(img, (260, 150), (45, 45), 0, 0, 360, (180, 30, 30), 2)
    cv2.putText(img, "SEAL", (238, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 30, 30), 2)
    cv2.line(img, (40, 165), (150, 135), (30, 30, 160), 2)
    cv2.line(img, (150, 135), (185, 160), (30, 30, 160), 2)

    img_path = tmp_path / "sig_stamp_doc.png"
    cv2.imwrite(str(img_path), img)
    return img, str(img_path)


def test_analyze_signature_stamp_workflow(tmp_path: Path) -> None:
    img, img_path = _make_doc_image(tmp_path)
    sig = inspect.signature(analyze_signature_stamp)
    params = list(sig.parameters.values())

    result = None
    for candidate in (img_path, img):
        try:
            if len(params) == 1:
                result = analyze_signature_stamp(candidate)
            else:
                result = analyze_signature_stamp(candidate)
            if result is not None:
                break
        except Exception:
            continue

    assert result is not None
