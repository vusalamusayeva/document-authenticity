"""
Document Type Classification (pipeline addım 3)
- sənəd növünün müəyyən edilməsi (ID kart, pasport, sürücülük vəsiqəsi, sertifikat və s.)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from doc_auth.utils.base_model import BaseDocModel


class DocumentClassifier(BaseDocModel):
    """configs/model.yaml -> document_classification parametrlərinə uyğun sənəd təsnifatçısı."""

    DEFAULT_CLASSES = [
        "id_card",
        "passport",
        "driver_license",
        "certificate",
        "generic_document",
    ]

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        classes = self.params.get("classes")
        if isinstance(classes, (list, tuple)) and len(classes) > 0:
            self.classes: List[str] = [str(c) for c in classes]
        else:
            self.classes = list(self.DEFAULT_CLASSES)
        self._model = None
        self._is_fitted = False

    def _resolve_image(self, item: Any) -> Optional[np.ndarray]:
        if isinstance(item, np.ndarray):
            return item if item.size > 0 else None
        if isinstance(item, dict):
            for key in ("image", "frame", "array", "data"):
                val = item.get(key)
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val
            doc_path = item.get("document_path") or item.get("path")
            if doc_path:
                return self._resolve_image(doc_path)
        if isinstance(item, (str, Path)):
            path = Path(item)
            if (
                path.exists()
                and path.is_file()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
            ):
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                return img if img is not None and img.size > 0 else None
        return None

    def _extract_feature_vector(self, item: Any) -> np.ndarray:
        """Girişdən (görüntü, fayl yolu və ya feature dict) sabit uzunluqlu xüsusiyyət vektoru çıxarır."""
        if isinstance(item, dict) and "features" in item and item["features"] is not None:
            arr = np.asarray(item["features"], dtype=np.float32).reshape(-1)
            if arr.size > 0 and np.all(np.isfinite(arr)):
                padded = np.zeros(4, dtype=np.float32)
                n = min(4, arr.size)
                padded[:n] = arr[:n]
                return padded

        image = self._resolve_image(item)
        if image is None:
            return np.zeros(4, dtype=np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        aspect_ratio = float(w) / float(max(1, h))
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            10,
        )
        text_density = float(np.count_nonzero(adaptive)) / float(max(1, w * h))
        edge_density = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / float(max(1, w * h))
        mean_intensity = float(np.mean(gray)) / 255.0

        return np.array(
            [aspect_ratio, text_density, edge_density, mean_intensity],
            dtype=np.float32,
        )

    def _score_row(self, feats: np.ndarray) -> np.ndarray:
        aspect_ratio, text_density, edge_density, mean_intensity = [float(x) for x in feats]
        n_cls = len(self.classes)
        scores = np.full(n_cls, 0.1, dtype=np.float64)

        def _set_if_present(label: str, val: float) -> None:
            if label in self.classes:
                scores[self.classes.index(label)] = max(0.01, val)

        # Kart / vəsiqə nisbəti (~1.4 - 1.85) və orta mətn sıxlığı
        if 1.35 <= aspect_ratio <= 1.85:
            _set_if_present("id_card", 0.72 + min(0.2, text_density * 2.0))
            _set_if_present("driver_license", 0.65 + min(0.2, edge_density * 2.0))
        # Pasport və ya portret/A4 sənəd nisbəti
        elif 0.65 <= aspect_ratio <= 1.35:
            _set_if_present("passport", 0.68 + min(0.2, mean_intensity * 0.2))
            _set_if_present("certificate", 0.64 + min(0.2, text_density * 1.5))
        else:
            _set_if_present("certificate", 0.66)
            _set_if_present("generic_document", 0.62 + min(0.2, text_density * 2.0))

        _set_if_present("generic_document", max(scores.max() * 0.5, 0.25))
        probs = scores / np.sum(scores)
        return probs

    def fit(self, X, y=None, **kwargs):
        if X is None:
            raise ValueError("fit() üçün X boş ola bilməz.")
        items = list(X) if not isinstance(X, np.ndarray) else [X[i] for i in range(len(X))]
        if len(items) == 0:
            raise ValueError("fit() üçün X boş siyahı ola bilməz.")
        if y is not None:
            labels = list(y) if not isinstance(y, np.ndarray) else [y[i] for i in range(len(y))]
            if len(labels) != len(items):
                raise ValueError("X və y uzunluqları uyğun olmalıdır.")
            for lbl in labels:
                lbl_str = str(lbl)
                if lbl_str not in self.classes:
                    self.classes.append(lbl_str)
        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if X is None:
            raise ValueError("predict_proba() üçün X boş ola bilməz.")
        if isinstance(X, np.ndarray) and X.ndim == 3:
            rows = [X]
        elif isinstance(X, dict):
            rows = [X]
        elif isinstance(X, (str, Path)):
            rows = [X]
        else:
            rows = list(X)

        if len(rows) == 0:
            return np.zeros((0, len(self.classes)), dtype=np.float64)

        proba_list: List[np.ndarray] = []
        for row in rows:
            feats = self._extract_feature_vector(row)
            proba_list.append(self._score_row(feats))
        return np.vstack(proba_list)

    def predict(self, X) -> List[str]:
        probs = self.predict_proba(X)
        if probs.shape[0] == 0:
            return []
        indices = np.argmax(probs, axis=1)
        return [self.classes[int(idx)] for idx in indices]

    def classify(self, payload: Any) -> Dict[str, Any]:
        image = self._resolve_image(payload)
        if image is None and not (isinstance(payload, dict) and "features" in payload):
            return {
                "status": "not_applicable",
                "document_type": None,
                "confidence": 0.0,
                "reason": "Sənəd təsnifatı üçün etibarlı görüntü və ya xüsusiyyət tapılmadı.",
            }
        probs = self.predict_proba([payload])[0]
        best_idx = int(np.argmax(probs))
        return {
            "status": "success",
            "document_type": self.classes[best_idx],
            "confidence": round(float(probs[best_idx]), 4),
            "probabilities": {
                cls_name: round(float(probs[i]), 4)
                for i, cls_name in enumerate(self.classes)
            },
        }