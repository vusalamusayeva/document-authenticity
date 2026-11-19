from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class OCRChecker:
    """
    Sənəd görüntüsündə mətn/OCR əlçatanlığını və əsas kontrast/mətn sıxlığı
    göstəricilərini real görüntü matrisi üzərindən yoxlayan modul.
    """

    def __init__(self, min_text_density: float = 0.002) -> None:
        if not (0.0 <= min_text_density <= 1.0):
            raise ValueError("min_text_density [0.0, 1.0] aralığında olmalıdır.")
        self.min_text_density = min_text_density

    def _resolve_image(self, payload: Any) -> Optional[np.ndarray]:
        if isinstance(payload, np.ndarray):
            return payload if payload.size > 0 else None
        if isinstance(payload, dict):
            for key in ("image", "frame", "array", "data"):
                val = payload.get(key)
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val
            doc_path = payload.get("document_path") or payload.get("path")
            if doc_path:
                return self._resolve_image(doc_path)
        if isinstance(payload, (str, Path)):
            path = Path(payload)
            if path.exists() and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                return img if img is not None and img.size > 0 else None
        return None

    def check(self, payload: Any) -> Dict[str, Any]:
        image = self._resolve_image(payload)
        if image is None:
            return {
                "status": "not_applicable",
                "has_text": None,
                "suspicious": None,
                "text_density": 0.0,
                "reason": "OCR analizi üçün görüntü matrisi tapılmadı.",
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        total_pixels = float(max(1, w * h))

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            10,
        )
        foreground_pixels = float(np.count_nonzero(adaptive))
        text_density = round(foreground_pixels / total_pixels, 6)
        has_text = bool(text_density >= self.min_text_density)

        return {
            "status": "success",
            "has_text": has_text,
            "suspicious": False,
            "text_density": text_density,
            "min_text_density": self.min_text_density,
        }

    def run(self, payload: Any) -> Dict[str, Any]:
        return self.check(payload)

    def analyze(self, payload: Any) -> Dict[str, Any]:
        return self.check(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.check(payload)