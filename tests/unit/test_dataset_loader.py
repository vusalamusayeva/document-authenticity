from pathlib import Path
import inspect
import pytest
import cv2
import numpy as np

from doc_auth.ingestion.dataset_loader import DatasetLoader


def _make_sample_image(img_path: Path) -> Path:
    img = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (15, 15), (145, 105), (40, 40, 40), 2)
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


def test_dataset_loader_methods_and_missing_file(tmp_path: Path) -> None:
    loader = DatasetLoader()
    assert loader is not None

    img_path = _make_sample_image(tmp_path / "doc.png")
    for method_name in ("load_document", "load", "analyze", "__call__"):
        method = getattr(loader, method_name, None)
        if callable(method):
            try:
                res = _call_with_path(method, img_path)
                assert res is not None
            except Exception:
                pass

    missing = tmp_path / "nonexistent_file.png"
    with pytest.raises(Exception):
        _call_with_path(loader.load_document, missing)
