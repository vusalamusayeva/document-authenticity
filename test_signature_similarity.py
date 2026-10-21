import numpy as np
import cv2
from doc_auth.signature_stamp.signature_similarity import SignatureSimilarity

def test_compare_identical_signatures():
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.putText(img1, "TestSig", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    result = SignatureSimilarity.compare(img1, img1)
    assert isinstance(result, dict)
    assert "total_similarity" in result
    assert result["total_similarity"] > 0.7

def test_compare_different_signatures():
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    
    result = SignatureSimilarity.compare(img1, img2)
    assert result["total_similarity"] < 0.5