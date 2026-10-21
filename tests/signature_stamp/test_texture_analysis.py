import numpy as np
from doc_auth.signature_stamp.texture_analysis import analyze_texture

def test_analyze_texture_keys_and_values():
    dummy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    result = analyze_texture(dummy_image)
    
    assert isinstance(result, dict)
    assert "sharpness_variance" in result
    assert "noise_difference" in result
    assert "digital_manipulation_probability" in result
    assert isinstance(result["sharpness_variance"], float)
    assert isinstance(result["digital_manipulation_probability"], float)
    assert 0.0 <= result["digital_manipulation_probability"] <= 1.0