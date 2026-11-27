import pytest

from doc_auth.fusion.decision_model import DecisionModel


def test_decision_model_threshold_validation() -> None:
    with pytest.raises(ValueError):
        DecisionModel(suspicious_threshold=-0.1)
    with pytest.raises(ValueError):
        DecisionModel(suspicious_threshold=1.5)


def test_decision_model_tampered_and_suspicious_signals() -> None:
    model = DecisionModel(suspicious_threshold=0.6)

    signals = {
        "forensics": {"tampered": True},
        "metadata": {"suspicious": True},
        "ocr": {"suspicious": True},
        "classification": {"status": "inconclusive"},
    }
    res = model.decide(signals)
    assert res["status"] == "success"
    assert res["decision"] == "suspicious"
    assert res["is_suspicious"] is True
    assert res["is_authentic"] is False
    assert res["risk_score"] == pytest.approx(0.925)
    assert len(res["contributions"]) == 4

    assert model.fuse(signals) == res
    assert model.predict(signals) == res
    assert model.analyze(signals) == res
    assert model(signals) == res
from doc_auth.fusion.decision_model import DecisionModel


def test_decide_returns_inconclusive_when_no_valid_signals() -> None:
    model = DecisionModel()
    result = model.decide({"metadata": {"status": "skipped"}, "ocr": {"status": "failed"}})
    assert result["status"] == "inconclusive"
    assert result["decision"] == "inconclusive"
    assert result["risk_score"] is None
