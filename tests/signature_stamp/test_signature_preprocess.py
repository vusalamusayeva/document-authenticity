import numpy as np
from doc_auth.signature_stamp.signature_preprocess import preprocess_signature

def test_preprocess_signature_real():
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    processed = preprocess_signature(dummy_image)
    
    assert processed is not None
    assert len(processed.shape) == 2
    assert processed.shape == (150, 300)