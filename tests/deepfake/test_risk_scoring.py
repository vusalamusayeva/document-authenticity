import pytest
from doc_auth.deepfake.exceptions import InvalidConfigError
from doc_auth.deepfake.risk_scoring import (
    DeepfakeRiskScorer,
    calculate_risk_score,
    validate_score_value,
)


def test_validate_score_and_weights_success_and_errors() -> None:
    assert validate_score_value(0.0) == 0.0
    assert validate_score_value(1) == 1.0
    assert validate_score_value(0.45, name="visual") == 0.45

    with pytest.raises(InvalidConfigError):
        validate_score_value(True, name="score")

    with pytest.raises(InvalidConfigError):
        validate_score_value("0.5", name="score")

    with pytest.raises(InvalidConfigError):
        validate_score_value(-0.1, name="score")

    with pytest.raises(InvalidConfigError):
        validate_score_value(1.2, name="score")

    with pytest.raises(InvalidConfigError):
        DeepfakeRiskScorer(visual_weight="0.5")  # type: ignore[arg-type]

    with pytest.raises(InvalidConfigError):
        DeepfakeRiskScorer(visual_weight=False)

    with pytest.raises(InvalidConfigError):
        DeepfakeRiskScorer(visual_weight=-0.2)

    with pytest.raises(InvalidConfigError):
        DeepfakeRiskScorer(visual_weight=0.0, temporal_weight=0.0, audio_weight=0.0)


def test_risk_scorer_signals_compute_risk_and_call() -> None:
    scorer = DeepfakeRiskScorer(suspicious_threshold=0.5)

    inconclusive = scorer({"visual": {"status": "failed", "score": None}})
    assert inconclusive["status"] == "inconclusive"
    assert inconclusive["decision"] == "inconclusive"
    assert inconclusive["risk_score"] is None

    res_explicit = scorer.compute_risk(
        max_visual_score=0.8,
        temporal_score=0.4,
        audio_analysis={"lip_sync_score": 0.6},
    )
    assert res_explicit["status"] == "success"
    assert "audio" in res_explicit["used_signals"]
    assert 0.0 <= res_explicit["overall_risk_score"] <= 1.0

    res_inferred_anomalous = scorer.compute_risk(
        max_visual_score=0.7,
        audio_analysis={"has_audio": True, "audio_status": "mismatch"},
    )
    assert res_inferred_anomalous["status"] == "success"
    assert "audio" in res_inferred_anomalous["used_signals"]

    res_inferred_normal = scorer.compute_risk(
        max_visual_score=0.2,
        audio_analysis={"has_audio": True, "audio_status": "clean"},
    )
    assert res_inferred_normal["status"] == "success"
    assert res_inferred_normal["decision"] == "low_risk"

    res_call = scorer({"visual": {"status": "success", "score": 0.2}})
    assert res_call["status"] == "success"
    assert res_call["decision"] == "low_risk"

    res_fn = calculate_risk_score(
        signals={"visual": {"status": "success", "score": 0.9}},
        weights={"visual": 1.0, "temporal": 0.0, "audio": 0.0},
        suspicious_threshold=0.5,
    )
    assert res_fn["status"] == "success"
    assert res_fn["is_suspicious"] is True
