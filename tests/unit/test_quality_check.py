import numpy as np
import pytest
from doc_auth.preprocessing.quality_check import check_blur, check_brightness, run_quality_check

def test_blur_check():
    # Kəskin (sharp) süni şəkil yaradırıq (ağ-qara xəttlər)
    sharp_image = np.zeros((300, 300), dtype=np.uint8)
    sharp_image[::10, :] = 255
    
    score, is_sharp = check_blur(sharp_image, threshold=10.0)
    assert isinstance(score, float)
    assert is_sharp is True

def test_brightness_check():
    # Normal parlaqlıqda boz şəkil
    normal_image = np.ones((100, 100), dtype=np.uint8) * 128
    
    score, is_acceptable = check_brightness(normal_image, min_b=40, max_b=220)
    assert isinstance(score, float)
    assert is_acceptable is True

def test_run_quality_check_pipeline():
    # Parlaqlığı və kəskinliyi normal olan rəngli şəkil yaradırıq
    image = np.ones((100, 100, 3), dtype=np.uint8) * 120
    image[::10, :, :] = 200
    
    cfg = {
        "blur_threshold": 0.0,
        "min_brightness": 40,
        "max_brightness": 220,
        "reject_on_low_quality": True
    }
    result = run_quality_check(image, cfg)
    assert result.is_acceptable is True