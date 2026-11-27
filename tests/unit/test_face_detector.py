from pathlib import Path
import cv2
import numpy as np
import pytest

from doc_auth.deepfake import face_detector as fd_mod
from doc_auth.deepfake.exceptions import ModelUnavailableError
from doc_auth.deepfake.face_detector import FaceDetector, detect_faces


class _DummyCascade:
    def __init__(self, boxes=None, empty: bool = False) -> None:
        self._boxes = boxes if boxes is not None else []
        self._empty = empty

    def empty(self) -> bool:
        return self._empty

    def detectMultiScale(self, gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        return self._boxes


def test_detect_faces_edge_cases_and_formats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert detect_faces(None) == []  # type: ignore[arg-type]
    assert detect_faces(np.array([], dtype=np.uint8)) == []
    assert detect_faces(np.zeros((10,), dtype=np.uint8)) == []
    assert detect_faces(np.zeros((0, 10, 3), dtype=np.uint8)) == []

    cascade_file = tmp_path / "haarcascade_frontalface_default.xml"
    cascade_file.write_text("<xml/>", encoding="utf-8")

    monkeypatch.setattr(cv2.data, "haarcascades", "", raising=False)
    monkeypatch.setattr(cv2, "__file__", str(tmp_path / "cv2" / "__init__.py"))
    (tmp_path / "cv2" / "data").mkdir(parents=True, exist_ok=True)
    fallback_xml = tmp_path / "cv2" / "data" / "haarcascade_frontalface_default.xml"
    fallback_xml.write_text("<xml/>", encoding="utf-8")
    assert fd_mod._resolve_cascade_path() == str(fallback_xml)

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda p: _DummyCascade(boxes=[(5, 6, 20, 20), (10, 10, 0, 10)]))

    rgba_float = np.full((40, 50, 4), 120.0, dtype=np.float32)
    dets = detect_faces(rgba_float)
    assert len(dets) == 1
    assert dets[0]["bbox"] == [5, 6, 25, 26]
    assert dets[0]["crop"].shape == (20, 20, 4)

    gray_img = np.full((30, 30), 100, dtype=np.uint8)
    assert len(detect_faces(gray_img)) == 1


def test_face_detector_wrapper_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cascade_file = tmp_path / "haarcascade_frontalface_default.xml"
    cascade_file.write_text("<xml/>", encoding="utf-8")
    monkeypatch.setattr(cv2.data, "haarcascades", str(tmp_path) + "/", raising=False)
    monkeypatch.setattr(cv2, "CascadeClassifier", lambda p: _DummyCascade(boxes=[(2, 2, 10, 10)]))

    detector = FaceDetector()
    assert detector.verify_model_weights() is True

    frame = np.zeros((25, 25, 3), dtype=np.uint8)
    assert len(detector.detect_faces(frame)) == 1
    assert len(detector.detect(frame)) == 1
    assert len(detector(frame)) == 1

    bad_detector = FaceDetector(model_path="non_existent_weights.pt")
    with pytest.raises(ModelUnavailableError):
        bad_detector.verify_model_weights()

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda p: _DummyCascade(empty=True))
    with pytest.raises(ModelUnavailableError):
        detector.verify_model_weights()

    monkeypatch.setattr(cv2.data, "haarcascades", str(tmp_path / "empty_dir") + "/", raising=False)
    monkeypatch.setattr(cv2, "__file__", str(tmp_path / "other" / "__init__.py"))
    with pytest.raises(ModelUnavailableError):
        fd_mod._resolve_cascade_path()
import numpy as np

from doc_auth.deepfake.face_detector import detect_faces


def test_detect_faces_returns_empty_for_zero_height_or_width() -> None:
    zero_h = np.empty((0, 10), dtype=np.uint8)
    assert detect_faces(zero_h) == []
import numpy as np
import pytest
from doc_auth.deepfake.face_detector import detect_faces


def test_detect_faces_returns_empty_when_zero_height_or_width() -> None:
    class _ZeroDimFrame(np.ndarray):
        @property
        def shape(self) -> tuple[int, int]:
            return (0, 2)

        @property
        def size(self) -> int:
            return 4

    dummy_frame = np.ones((2, 2), dtype=np.uint8).view(_ZeroDimFrame)
    assert detect_faces(dummy_frame) == []
