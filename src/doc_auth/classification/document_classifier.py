"""
Document Type Classification (pipeline add?m 3)
- s?n?d n?v?n?n m??yy?n edilm?si (ID kart, pasport, s?r?c?l?k v?siq?si, sertifikat, unknown)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from doc_auth.utils.base_model import BaseDocModel


class DocumentClassifier(BaseDocModel):
    """Real ML t?snifat??s? v? etibarl?l?q (confidence) kalibrasiyas? olan s?n?d t?snifat??s?."""

    DEFAULT_CLASSES = [
        "id_card",
        "passport",
        "drivers_license",
        "certificate",
        "unknown",
    ]

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        classes = self.params.get("classes")
        if isinstance(classes, (list, tuple)) and len(classes) > 0:
            self.classes: List[str] = [str(c) for c in classes]
        else:
            self.classes = list(self.DEFAULT_CLASSES)
        if "unknown" not in self.classes:
            self.classes.append("unknown")

        conf_val = self.params.get("min_confidence", 0.45)
        self.min_confidence: float = float(conf_val)
        self._scaler: Optional[StandardScaler] = None
        self._clf: Optional[LogisticRegression] = None
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
        """Giri?d?n (g?r?nt?, fayl yolu v? ya feature dict) sabit uzunluqlu x?susiyy?t vektoru ??xar?r."""
        if isinstance(item, dict) and "features" in item and item["features"] is not None:
            arr = np.asarray(item["features"], dtype=np.float32).reshape(-1)
            if arr.size > 0 and np.all(np.isfinite(arr)):
                out = np.zeros(8, dtype=np.float32)
                n = min(8, arr.size)
                out[:n] = arr[:n]
                return out

        image = self._resolve_image(item)
        if image is None:
            return np.zeros(8, dtype=np.float32)

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
        std_intensity = float(np.std(gray)) / 255.0

        top_half = gray[: max(1, h // 2), :]
        bot_half = gray[max(1, h // 2) :, :]
        top_text = float(np.count_nonzero(adaptive[: max(1, h // 2), :])) / float(max(1, top_half.size))
        bot_text = float(np.count_nonzero(adaptive[max(1, h // 2) :, :])) / float(max(1, bot_half.size))
        symmetry = 1.0 - min(1.0, abs(top_text - bot_text))

        return np.array(
            [
                aspect_ratio,
                text_density,
                edge_density,
                mean_intensity,
                std_intensity,
                top_text,
                bot_text,
                symmetry,
            ],
            dtype=np.float32,
        )

    def fit(self, X, y=None, **kwargs):
        if X is None:
            raise ValueError("fit() ???n X bo? ola bilm?z.")
        items = list(X) if not isinstance(X, np.ndarray) else [X[i] for i in range(len(X))]
        if len(items) == 0:
            raise ValueError("fit() ???n X bo? siyah? ola bilm?z.")
        if y is None:
            raise ValueError("fit() ???n y etiketl?ri t?l?b olunur.")

        labels = [str(lbl) for lbl in (list(y) if not isinstance(y, np.ndarray) else [y[i] for i in range(len(y))])]
        if len(labels) != len(items):
            raise ValueError("X v? y uzunluqlar? uy?un olmal?d?r.")

        unique_labels = list(dict.fromkeys(labels))
        if len(unique_labels) < 2:
            raise ValueError("Model ?yr?dilm?si ???n ?n az? 2 f?rqli sinif t?l?b olunur.")

        for lbl in unique_labels:
            if lbl not in self.classes:
                self.classes.append(lbl)

        feats = np.vstack([self._extract_feature_vector(item) for item in items])
        scaler = StandardScaler()
        scaled_x = scaler.fit_transform(feats)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(scaled_x, labels)

        self._scaler = scaler
        self._clf = clf
        self.classes = [str(c) for c in clf.classes_]
        self._is_fitted = True
        return self

    def _ensure_fitted(self) -> None:
        if not self._is_fitted or self._clf is None or self._scaler is None:
            raise RuntimeError(
                "DocumentClassifier ?yr?dilm?yib. ?vv?lc? fit(X, y) ?a??r?n v? ya ?yr?dilmi? model y?kl?yin."
            )

    def predict_proba(self, X) -> np.ndarray:
        self._ensure_fitted()
        assert self._clf is not None and self._scaler is not None

        if X is None:
            raise ValueError("predict_proba() ???n X bo? ola bilm?z.")
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

        feats = np.vstack([self._extract_feature_vector(row) for row in rows])
        scaled_x = self._scaler.transform(feats)
        return np.asarray(self._clf.predict_proba(scaled_x), dtype=np.float64)

    def predict(self, X) -> List[str]:
        probs = self.predict_proba(X)
        if probs.shape[0] == 0:
            return []
        predictions: List[str] = []
        for row_probs in probs:
            best_idx = int(np.argmax(row_probs))
            best_conf = float(row_probs[best_idx])
            if best_conf < self.min_confidence:
                predictions.append("unknown")
            else:
                predictions.append(self.classes[best_idx])
        return predictions

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        self._ensure_fitted()
        y_true = [str(v) for v in y_test]
        y_pred = self.predict(X_test)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=self.classes,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        per_class = {
            cls_name: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, cls_name in enumerate(self.classes)
        }
        return {
            "labels": list(self.classes),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }

    def classify(self, payload: Any) -> Dict[str, Any]:
        image = self._resolve_image(payload)
        if image is None and not (isinstance(payload, dict) and "features" in payload):
            return {
                "status": "not_applicable",
                "document_type": None,
                "confidence": 0.0,
                "reason": "S?n?d t?snifat? ???n etibarl? g?r?nt? v? ya x?susiyy?t tap?lmad?.",
            }
        if not self._is_fitted or self._clf is None or self._scaler is None:
            return {
                "status": "inconclusive",
                "document_type": "unknown",
                "confidence": 0.0,
                "reason": "S?n?d t?snifat modeli h?l? ?yr?dilm?yib (fit olunmay?b).",
            }

        probs = self.predict_proba([payload])[0]
        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        prob_map = {
            cls_name: round(float(probs[i]), 4)
            for i, cls_name in enumerate(self.classes)
        }

        if best_conf < self.min_confidence:
            return {
                "status": "inconclusive",
                "document_type": "unknown",
                "confidence": round(best_conf, 4),
                "reason": f"Maksimum sinif etibarl?l??? ({best_conf:.4f}) h?dd?n ({self.min_confidence:.4f}) a?a??d?r.",
                "probabilities": prob_map,
            }

        return {
            "status": "success",
            "document_type": self.classes[best_idx],
            "confidence": round(best_conf, 4),
            "probabilities": prob_map,
        }

    def save(self, path: str) -> None:
        import pickle
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DocumentClassifier":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"G?zl?nil?n tip {cls.__name__}, al?nd?: {type(obj).__name__}")
        return obj

