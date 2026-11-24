import cv2
import numpy as np
from typing import Dict, Any

class DeepfakeVisualizer:
    def __init__(self):
        pass

    def draw_annotations(self, frame: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        annotated = frame.copy()
        faces = analysis_result.get("faces", [])

        for face in faces:
            bbox = face.get("bbox", [])
            score = face.get("deepfake_score", 0.0)
            if len(bbox) == 4:
                x, y, w, h = bbox
                color = (0, 0, 255) if score > 0.5 else (0, 255, 0)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    annotated, 
                    f"Score: {score:.2f}", 
                    (x, max(0, y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )

        return annotated