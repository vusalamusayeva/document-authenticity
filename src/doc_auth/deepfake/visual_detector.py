import numpy as np
from typing import Dict, Any
from doc_auth.deepfake.face_detector import FaceDetector

class VisualDetector:
    def __init__(self, face_detector: FaceDetector = None):
        self.face_detector = face_detector or FaceDetector()

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        faces = self.face_detector.detect_faces(frame)
        max_score = 0.0
        face_results = []

        for face in faces:
            score = 0.15
            max_score = max(max_score, score)
            face_results.append({
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "deepfake_score": score
            })

        return {
            "face_detected": len(faces) > 0,
            "faces": face_results,
            "max_deepfake_score": max_score
        }