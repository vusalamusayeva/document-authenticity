import builtins
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import pytest

from doc_auth.ocr.ocr_check import OCRChecker


def test_ocr_checker_init_validation() -> None:
    with pytest.raises(ValueError):
        OCRChecker(min_text_density=-0.01)
    with pytest.raises(ValueError):
        OCRChecker(min_text_density=1.1)
    with pytest.raises(ValueError):
        OCRChecker(min_confidence=-0.1)
    with pytest.raises(ValueError):
        OCRChecker(min_confidence=1.2)


def test_ocr_checker_resolve_image_and_not_applicable(tmp_path: Path) -> None:
    checker = OCRChecker()

    assert checker._resolve_image(np.array([], dtype=np.uint8)) is None

    direct_arr = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert checker._resolve_image(direct_arr) is direct_arr
    assert checker._resolve_image({"frame": direct_arr}) is direct_arr

    img_path = tmp_path / "sample.png"
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.putText(img, "ABC", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    assert cv2.imwrite(str(img_path), img)

    resolved = checker._resolve_image({"document_path": str(img_path)})
    assert isinstance(resolved, np.ndarray) and resolved.size > 0

    none_res = checker.check({"path": str(tmp_path / "missing.png")})
    assert none_res["status"] == "not_applicable"
    assert none_res["has_text"] is None


def test_ocr_checker_tesseract_missing_and_easyocr_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = OCRChecker(min_text_density=0.0, min_confidence=0.5, lang="eng")
    gray = np.full((40, 120), 255, dtype=np.uint8)
    cv2.putText(gray, "TEST", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    fake_pytesseract = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="nonexistent_tesseract_bin"),
        Output=types.SimpleNamespace(DICT="dict"),
        image_to_data=lambda *a, **kw: {},
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    monkeypatch.setattr("shutil.which", lambda exe: None)

    class _FakeReader:
        def __init__(self, langs, gpu=False, verbose=False) -> None:
            self.langs = langs

        def readtext(self, arr):
            return [([[0, 0], [10, 0], [10, 10], [0, 10]], "FALLBACK", 0.9)]

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=_FakeReader))

    res = checker.check(gray)
    assert res["status"] == "success"
    assert res["engine"] == "easyocr"
    assert res["text"] == "FALLBACK"
    assert res["suspicious"] is False
    assert checker.run(gray) == res
    assert checker.analyze(gray) == res
    assert checker(gray) == res

    def _fail_import(name, *args, **kwargs):
        if name in {"pytesseract", "easyocr"}:
            raise ImportError(f"no {name}")
        return orig_import(name, *args, **kwargs)

    orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _fail_import)
    unavail = checker.check(gray)
    assert unavail["status"] == "ocr_unavailable"
    assert "error" in unavail


def test_ocr_checker_tesseract_invalid_conf_and_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = OCRChecker(min_text_density=0.0, min_confidence=0.8)
    gray = np.full((32, 100), 255, dtype=np.uint8)
    cv2.putText(gray, "LOW", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    fake_pytesseract = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="tesseract"),
        Output=types.SimpleNamespace(DICT="dict"),
        image_to_data=lambda *a, **kw: {
            "text": ["", "LOW", "BADCONF"],
            "conf": [-1, "40", "not_a_number"],
            "left": [0, 5, 40],
            "top": [0, 5, 5],
            "width": [0, 30, 40],
            "height": [0, 20, 20],
        },
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    monkeypatch.setattr("shutil.which", lambda exe: "C:/fake/tesseract.exe")

    res = checker.check(gray)
    assert res["status"] == "success"
    assert res["engine"] == "tesseract"
    assert res["text"] == "LOW BADCONF"
    assert res["has_text"] is True
    assert res["suspicious"] is True


def test_run_easyocr_skips_empty_text_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeReader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def readtext(self, _gray: np.ndarray) -> list[tuple[object, str, float]]:
            return [([[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.9)]

    fake_easyocr = types.ModuleType("easyocr")
    fake_easyocr.Reader = _FakeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    checker = OCRChecker()
    out = checker._run_easyocr(np.zeros((20, 20), dtype=np.uint8))
    assert out["engine"] == "easyocr"
    assert out["text"] == ""
    assert out["fields"] == []
