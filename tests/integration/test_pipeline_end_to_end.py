import numpy as np
import pytest
from doc_auth.preprocessing.quality_check import run_quality_check

def test_pipeline_end_to_end_basic():
    """
    End-to-end inteqrasiya testi: 
    Keyfiyyət yoxlaması (preprocessing) ilə başlanğıc pipeline axınını simulyasiya edir.
    """
    # Süni sənəd şəkli yaradırıq (normal parlaqlıq və kəskinliklə)
    image = np.ones((200, 200, 3), dtype=np.uint8) * 150
    image[::20, :, :] = 220
    
    # Pipeline konfiqurasiyası
    cfg = {
        "blur_threshold": 0.0,
        "min_brightness": 40,
        "max_brightness": 240,
        "reject_on_low_quality": True
    }
    
    # İcra mərhələsi: Quality check
    quality_result = run_quality_check(image, cfg)
    
    # Yoxlama (Assertions)
    assert quality_result.is_acceptable is True
    assert quality_result.blur_score >= 0.0
    assert 40 <= quality_result.brightness_score <= 240