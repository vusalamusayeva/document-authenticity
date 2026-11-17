import pytest
from doc_auth.deepfake.face_detector import FaceDetector
from doc_auth.deepfake.exceptions import ModelUnavailableError
import numpy as np

def test_face_detector_missing_weights():
    detector = FaceDetector(model_path="non_existent_weights.pth")
    with pytest.raises(ModelUnavailableError):
        detector.verify_model_weights()

def test_face_detector_basic():
    detector = FaceDetector()
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detector.detect_faces(dummy_frame)
    assert isinstance(faces, list)