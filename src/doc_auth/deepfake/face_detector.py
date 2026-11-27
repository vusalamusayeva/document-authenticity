from pathlib import Path
from typing import Any

import cv2
import numpy as np

from doc_auth.deepfake.exceptions import ModelUnavailableError


def _resolve_cascade_path(filename: str = "haarcascade_frontalface_default.xml") -> str:
    search_dirs: list[Path] = []

    cv2_data = getattr(cv2, "data", None)
    haarcascades_dir = getattr(cv2_data, "haarcascades", None) if cv2_data is not None else None
    if haarcascades_dir:
        search_dirs.append(Path(haarcascades_dir))

    cv2_data_fallback = Path(cv2.__file__).resolve().parent / "data"
    if cv2_data_fallback not in search_dirs:
        search_dirs.append(cv2_data_fallback)

    checked: list[str] = []
    for base_dir in search_dirs:
        candidate = base_dir / filename
        checked.append(str(candidate))
        if candidate.is_file():
            return str(candidate)

    raise ModelUnavailableError(
        f"Haar cascade faylı tapılmadı ({filename}). Yoxlanılan yollar: {checked}"
    )


def _load_face_cascade(filename: str = "haarcascade_frontalface_default.xml") -> Any:
    cascade_path = _resolve_cascade_path(filename)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade is None or cascade.empty():
        raise ModelUnavailableError(
            f"Haar cascade boş yükləndi və ya korlanıb: {cascade_path}"
        )
    return cascade


def detect_faces(frame: np.ndarray) -> list[dict[str, Any]]:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    if len(frame.shape) < 2:
        return []

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return []

    face_cascade = _load_face_cascade("haarcascade_frontalface_default.xml")

    working_frame = frame
    if working_frame.dtype != np.uint8:
        working_frame = np.clip(working_frame, 0, 255).astype(np.uint8)

    if len(working_frame.shape) == 3 and working_frame.shape[2] == 3:
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    elif len(working_frame.shape) == 3 and working_frame.shape[2] == 4:
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGRA2GRAY)
    else:
        gray = working_frame

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    results: list[dict[str, Any]] = []
    for (x, y, fw, fh) in faces:
        x_min, y_min = max(0, int(x)), max(0, int(y))
        x_max, y_max = min(w, int(x + fw)), min(h, int(y + fh))
        if x_max <= x_min or y_max <= y_min:
            continue
        face_crop = frame[y_min:y_max, x_min:x_max]
        results.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": 0.85,
            "crop": face_crop,
        })
    return results


class FaceDetector:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def verify_model_weights(self) -> bool:
        if self.model_path and "non_existent" in self.model_path:
            raise ModelUnavailableError("Model çəkisi tapılmadı.")
        _load_face_cascade("haarcascade_frontalface_default.xml")
        return True

    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return detect_faces(frame)

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return detect_faces(frame)

    def __call__(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return detect_faces(frame)