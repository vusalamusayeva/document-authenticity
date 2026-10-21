import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from .schemas import BoundingBox
from .exceptions import SignatureStampError

class SignatureStampDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        self.threshold = confidence_threshold

    def _color_masking(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return blue_mask

    def detect_stamps_and_signatures(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        try:
            if image is None or image.size == 0:
                raise SignatureStampError("Analiz üçün şəkil tapılmadı.")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )

            if len(image.shape) == 3:
                blue_mask = self._color_masking(image)
                thresh = cv2.bitwise_or(thresh, blue_mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            stamps = []
            signatures = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                x, y, sw, sh = cv2.boundingRect(cnt)
                aspect_ratio = float(sw) / sh if sh > 0 else 0

                if 0.7 <= aspect_ratio <= 1.3 and 1000 < area < 50000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if circularity > self.threshold:
                            stamps.append({
                                "bbox": BoundingBox(ymin=y, xmin=x, ymax=y+sh, xmax=x+sw),
                                "type": "stamp",
                                "confidence": float(circularity)
                            })

                elif aspect_ratio > 1.2 or aspect_ratio < 0.8:
                    if area > 800:
                        signatures.append({
                            "bbox": BoundingBox(ymin=y, xmin=x, ymax=y+sh, xmax=x+sw),
                            "type": "signature",
                            "confidence": 0.75
                        })

            return {"stamps": stamps, "signatures": signatures}

        except Exception as e:
            raise SignatureStampError(f"Deteksiya zamanı xəta: {str(e)}")

    def detect_signature(self, image: np.ndarray) -> Optional[BoundingBox]:
        results = self.detect_stamps_and_signatures(image)
        sigs = results.get("signatures", [])
        if sigs:
            return sigs[0]["bbox"]
        return None