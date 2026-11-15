import pytest
import numpy as np
from src.doc_auth.deepfake.face_detector import FaceDetector
from src.doc_auth.deepfake.exceptions import ModelUnavailableError

def test_face_detector_missing_weights():
    detector = FaceDetector(model_path="non_existent_weights.pth")
    with pytest.raises(ModelUnavailableError):
        detector.verify_model_weights()

def test_face_detector_basic():
    detector = FaceDetector()
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detector.detect_faces(dummy_frame)
    assert isinstance(faces, list)