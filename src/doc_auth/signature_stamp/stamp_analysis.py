import cv2
import numpy as np
from .schemas import StampResult
from .exceptions import StampAnalysisError

def analyze_stamp(stamp_image: np.ndarray) -> StampResult:
    try:
        if stamp_image is None or stamp_image.size == 0:
            return StampResult(detected=False, confidence=0.0, analysis_details="Möhür şəkli tapılmadı.", circularity=0.0, solidity=0.0)

        gray = cv2.cvtColor(stamp_image, cv2.COLOR_BGR2GRAY) if len(stamp_image.shape) == 3 else stamp_image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return StampResult(detected=False, confidence=0.0, analysis_details="Möhür konturu tapılmadı.", circularity=0.0, solidity=0.0)

        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        perimeter = cv2.arcLength(largest_cnt, True)

        if perimeter == 0 or area < 200:
            return StampResult(detected=False, confidence=0.1, analysis_details="Obyekt çox kiçikdir.", circularity=0.0, solidity=0.0)

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        hull = cv2.convexHull(largest_cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        color_score = 1.0
        if len(stamp_image.shape) == 3:
            b, g, r = cv2.split(stamp_image)
            blue_dominance = np.mean(b) - np.mean(r)
            color_score = float(np.clip(blue_dominance / 50.0, 0.2, 1.0))

        confidence = float((circularity * 0.4) + (solidity * 0.4) + (color_score * 0.2))
        detected = confidence > 0.4

        details = (
            f"Dairəvilik: {circularity:.2f}, Solidity: {solidity:.2f}, "
            f"Rəng: {color_score:.2f}. {'Möhür normaldır.' if detected else 'Möhür şübhəlidir.'}"
        )

        return StampResult(
            detected=detected,
            confidence=max(0.0, min(1.0, confidence)),
            analysis_details=details,
            circularity=float(circularity),
            solidity=float(solidity)
        )

    except Exception as e:
        raise StampAnalysisError(f"Mühür analizi xətası: {str(e)}")