from doc_auth.explainability.explain import Explainer


def test_explainer_signals_factors_and_summary_branches() -> None:
    explainer = Explainer()

    # Empty payload -> default summary without risk_score
    empty_res = explainer.explain(None)
    assert empty_res["status"] == "success"
    assert empty_res["decision"] == "evaluated"
    assert empty_res["risk_score"] is None
    assert "evaluated" in empty_res["summary"]
    assert empty_res["factors"] == []

    # Rich payload with decision contributions, risk_score, and all signal types
    rich_payload = {
        "decision": {
            "decision": "review",
            "risk_score": 0.72,
            "contributions": [
                {"signal": "forensics", "weight": 0.4, "value": 0.8, "weighted_risk": 0.32},
                "invalid_item",
            ],
        },
        "signals": {
            "forensics": {"tampering_score": 0.8, "tampered": True},
            "metadata": {
                "suspicious_signals": [
                    {"field": "Producer", "reason": "PDF editor detected"},
                    {"field": "Author", "reason": ""},
                ]
            },
            "ocr": {"status": "ok", "suspicious": False},
            "classification": {"status": "ok", "document_type": "passport"},
        },
    }

    res = explainer.explain(rich_payload)
    assert res["decision"] == "review"
    assert res["risk_score"] == 0.72
    assert "0.72" in res["summary"]
    assert len(res["factors"]) == 1
    assert res["factors"][0]["signal"] == "forensics"
    assert any("Forensics ELA skoru" in h for h in res["evidence_highlights"])
    assert any("Metadata [Producer]" in h for h in res["evidence_highlights"])
    assert any("OCR status" in h for h in res["evidence_highlights"])
    assert any("Classification:" in h for h in res["evidence_highlights"])

    # Explicit summary override + analyze / __call__ aliases
    custom_payload = {
        "decision": {"status": "reject", "summary": "Explicit summary text"},
        "signals": {},
    }
    assert explainer.analyze(custom_payload)["summary"] == "Explicit summary text"
    assert explainer(custom_payload)["decision"] == "reject"
