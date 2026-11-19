from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class DocumentDetector:
    """
    Yüklənmiş görüntü və ya sənəd məlumatından sənəd sərhədi/bölgəsini müəyyən edən modul.
    """

    def __init__(self, min_area_ratio: float = 0.05) -> None:
        if not (0.0 < min_area_ratio < 1.0):
            raise ValueError("min_area_ratio (0.0, 1.0) aralığında olmalıdır.")
        self.min_area_ratio = min_area_ratio

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

    def detect(self, payload: Any) -> Dict[str, Any]:
        image = self._resolve_image(payload)
        if image is None:
            passthrough = dict(payload) if isinstance(payload, dict) else {"value": payload}
            passthrough.setdefault("status", "passthrough")
            passthrough.setdefault("detected", None)
            return passthrough

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        img_area = float(w * h)
        best_bbox = [0, 0, w, h]
        best_area = img_area
        found_contour = False

        for cnt in contours or []:
            area = float(cv2.contourArea(cnt))
            if area >= img_area * self.min_area_ratio:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > 0 and ch > 0:
                    best_bbox = [int(x), int(y), int(cw), int(ch)]
                    best_area = area
                    found_contour = True
                    break

        result: Dict[str, Any] = {
            "status": "success",
            "detected": True,
            "has_contour": found_contour,
            "bbox": best_bbox,
            "area_ratio": round(best_area / img_area, 4) if img_area > 0 else 0.0,
            "width": int(w),
            "height": int(h),
            "image": image,
        }
        if isinstance(payload, dict):
            for key in ("path", "document_path", "file_name", "extension"):
                if key in payload:
                    result[key] = payload[key]
        return result

    def detect_document(self, payload: Any) -> Dict[str, Any]:
        return self.detect(payload)

    def analyze(self, payload: Any) -> Dict[str, Any]:
        return self.detect(payload)

    def __call__(self, payload: Any) -> Dict[str, Any]:
        return self.detect(payload)