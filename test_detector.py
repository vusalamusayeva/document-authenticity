import numpy as np
import cv2
from doc_auth.signature_stamp.detector import SignatureStampDetector

def test_detect_signature_positive():
    detector = SignatureStampDetector()
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    # İmza oxşar qara xətt çəkilir
    cv2.line(img, (50, 150), (250, 150), (0, 0, 0), 3)
    
    boxes = detector.detect_stamps_and_signatures(img)
    assert "signatures" in boxes
    assert len(boxes["signatures"]) > 0

def test_detect_stamp_positive():
    detector = SignatureStampDetector()
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    # Dairəvi möhür çəkilir
    cv2.circle(img, (150, 150), 60, (255, 0, 0), 4)
    
    boxes = detector.detect_stamps_and_signatures(img)
    assert "stamps" in boxes
    assert len(boxes["stamps"]) > 0