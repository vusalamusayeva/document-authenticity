from pathlib import Path
import inspect
import cv2
import numpy as np

from doc_auth import service as svc_mod
from doc_auth.service import AnalysisService


def _make_doc_image(img_path: Path) -> Path:
    img = np.full((180, 240, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (220, 160), (30, 30, 30), 2)
    cv2.putText(img, "SAMPLE DOC", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.imwrite(str(img_path), img)
    return img_path


def test_service_init_sanitize_and_analyze(tmp_path: Path) -> None:
    sanitized = svc_mod._sanitize_for_json({
        "arr": np.array([1, 2], dtype=np.int64),
        "path": tmp_path / "doc.png",
        "nested": {"flag": np.bool_(True)},
    })
    assert sanitized["arr"] == [1, 2]
    assert sanitized["nested"]["flag"] is True

    service = AnalysisService()
    assert service is not None

    img_path = _make_doc_image(tmp_path / "doc.png")
    sig = inspect.signature(service.analyze)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if params:
        try:
            res = service.analyze(img_path)
        except TypeError:
            res = service.analyze(str(img_path))
        assert res is not None
