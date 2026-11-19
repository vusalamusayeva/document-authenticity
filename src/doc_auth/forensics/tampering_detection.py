from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class TamperingDetector:
    """
    Sənəd və ya görüntüdə saxtalaşdırma / redaktə əlamətlərini (ELA / kənar və səs-küy anomaliyaları)
    real görüntü matrisi üzərindən yoxlayan detektor.
    """

    def __init__(self, anomaly_threshold: float = 0.65, jpeg_quality: int = 90) -> None:
        if not (0.0 <= anomaly_threshold <= 1.0):
            raise ValueError("anomaly_threshold [0.0, 1.0] aralığında olmalıdır.")
        if not (1 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality [1, 100] aralığında olmalıdır.")
        self.anomaly_threshold = anomaly_threshold
        self.jpeg_quality = jpeg_quality

    def _extract_image(self, payload: Any) -> Optional[np.ndarray]:
        if isinstance(payload, np.ndarray):
            return payload if payload.size > 0 else None
        if isinstance(payload, dict):
            for key in ("image", "frame", "array", "data"):
                val = payload.get(key)
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val
            doc_path = payload.get("document_path") or payload.get("path")
            if doc_path:
                return self._extract_image(doc_path)
        if isinstance(payload, (str, Path)):
            path = Path(payload)
            if path.exists() and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                return img if img is not None and img.size > 0 else None
        return None

    def detect(self, payload: Any) -> Dict[str, Any]:
        image = self._extract_image(payload)
        if image is None:
            return {
                "status": "not_applicable",
                "tampered": None,
                "tampering_score": None,
                "reason": "Görüntü matrisi tapılmadı və ya dəstəklənməyən formatdır.",
            }

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        ok, encoded = cv2.imencode(".jpg", image, encode_params)
        if not ok:
            return {
                "status": "error",
                "tampered": None,
                "tampering_score": None,
                "reason": "Görüntü yenidən kodlaşdırıla bilmədi.",
            }

        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if recompressed is None or recompressed.shape != image.shape:
            return {
                "status": "error",
                "tampered": None,
                "tampering_score": None,
                "reason": "Yenidən sıxılmış görüntü dekod edilə bilmədi.",
            }

        diff = np.abs(image.astype(np.float32) - recompressed.astype(np.float32))
        mean_diff = float(np.mean(diff))
        p95_diff = float(np.percentile(diff, 95))

        # Orta və 95-ci persentil fərqini [0.0, 1.0] şkalasına normallaşdırırıq
        score = float(max(0.0, min(1.0, (0.6 * mean_diff + 0.4 * p95_diff) / 32.0)))
        tampered = bool(score >= self.anomaly_threshold)

        return {
            "status": "success",
            "tampered": tampered,
            "tampering_score": score,
            "mean_ela_diff": mean_diff,
            "p95_ela_diff": p95_diff,
        }

    def analyze(self, payload: Any) -> Dict[str, Any]:
        return self.detect(payload)

    def run(self, payload: Any) -> Dict[str, Any]:
        return self.detect(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.detect(payload)