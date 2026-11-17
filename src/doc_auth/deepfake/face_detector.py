import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
from doc_auth.deepfake.exceptions import ModelUnavailableError

class FaceDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def verify_model_weights(self):
        if self.model_path and not os.path.exists(self.model_path):
            raise ModelUnavailableError(f"Model checkpoint tapılmadı: {self.model_path}")

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detected_faces = []
        for (x, y, w, h) in faces:
            detected_faces.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": 0.95
            })
        return detected_faces