import pytest
import os
import numpy as np
from doc_auth.deepfake.pipeline import analyze_video_deepfake
from doc_auth.deepfake.risk_scoring import calculate_risk_score
from doc_auth.deepfake.exceptions import InvalidConfigError

def test_risk_scoring_validation():
    """Çəkilərin cəminin mənfi və ya sıfır olmasını yoxlayır."""
    with pytest.raises(InvalidConfigError):
        calculate_risk_score({"visual": {"status": "success", "score": 0.5}}, {"visual": 0.0})

def test_pipeline_with_invalid_bytes():
    """Etibarsız və ya boş video bayitləri ilə pipeline-ın xəta qaytarmasını yoxlayır."""
    result = analyze_video_deepfake(b"fake_video_bytes_content")
    assert result["status"] == "failed"
    assert "error_type" in result