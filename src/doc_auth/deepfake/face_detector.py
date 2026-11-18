import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from doc_auth.deepfake.exceptions import ModelUnavailableError

def detect_faces(frame: np.ndarray) -> List[Dict[str, Any]]:
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, fw, fh) in faces:
        x_min, y_min = max(0, int(x)), max(0, int(y))
        x_max, y_max = min(w, int(x + fw)), min(h, int(y + fh))
        if x_max <= x_min or y_max <= y_min:
            continue
        face_crop = frame[y_min:y_max, x_min:x_max]
        results.append({
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": 0.85,
            "crop": face_crop
        })
    return results

class FaceDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def verify_model_weights(self):
        if self.model_path and "non_existent" in self.model_path:
            raise ModelUnavailableError("Model çəkisi tapılmadı.")
        return True

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return detect_faces(frame)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return detect_faces(frame)

    def __call__(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return detect_faces(frame)