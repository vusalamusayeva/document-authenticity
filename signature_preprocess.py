import cv2
import numpy as np
from .exceptions import SignatureStampError

def preprocess_signature(image: np.ndarray, bbox: tuple = None) -> np.ndarray:
    try:
        if image is None or image.size == 0:
            raise SignatureStampError("Giriş şəkli boşdur.")

        img_copy = image.copy()
        if bbox is not None:
            ymin, xmin, ymax, xmax = bbox
            img_cropped = img_copy[ymin:ymax, xmin:xmax]
        else:
            img_cropped = img_copy

        if img_cropped.size == 0:
            img_cropped = img_copy

        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY) if len(img_cropped.shape) == 3 else img_cropped
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            signature_roi = thresh[y:y+h, x:x+w]
        else:
            signature_roi = thresh

        target_h, target_w = 150, 300
        h_roi, w_roi = signature_roi.shape
        if h_roi == 0 or w_roi == 0:
            return np.ones((target_h, target_w), dtype=np.uint8) * 255

        scale = min(target_w / w_roi, target_h / h_roi)
        new_w = int(w_roi * scale)
        new_h = int(h_roi * scale)

        resized = cv2.resize(signature_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        standard_img = np.ones((target_h, target_w), dtype=np.uint8) * 255

        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        standard_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 255 - resized

        return standard_img

    except Exception as e:
        raise SignatureStampError(f"İmza ön işlənməsi zamanı xəta: {str(e)}")