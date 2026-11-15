import pytest
from src.doc_auth.deepfake.risk_scoring import RiskScorer

def test_risk_scorer_calculation():
    scorer = RiskScorer(visual_weight=0.5, temporal_weight=0.3, audio_weight=0.2)
    audio_analysis = {"has_audio": True, "audio_status": "normal"}
    
    result = scorer.compute_risk(max_visual_score=0.8, temporal_score=0.6, audio_analysis=audio_analysis)
    
    assert "overall_risk_score" in result
    assert "is_suspicious" in result
    assert 0.0 <= result["overall_risk_score"] <= 1.0