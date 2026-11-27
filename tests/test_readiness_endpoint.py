from fastapi.testclient import TestClient
from doc_auth import api as doc_api_mod
from doc_auth.api import create_app


class _ReadyService:
    def analyze_summary(self, path):
        return {"verdict": "authentic", "confidence": 0.9}


def test_ready_returns_503_when_ocr_or_model_missing(monkeypatch):
    monkeypatch.setattr(doc_api_mod, "check_readiness_state", lambda svc: (False, {"status": "not_ready", "model_ready": True, "ocr_ready": False, "missing_langs": ["aze"]}))
    client = TestClient(create_app(analysis_service=_ReadyService()))
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["detail"]["status"] == "not_ready"


def test_ready_returns_200_when_model_and_ocr_ready(monkeypatch):
    monkeypatch.setattr(doc_api_mod, "check_readiness_state", lambda svc: (True, {"status": "ready", "model_ready": True, "ocr_ready": True, "missing_langs": []}))
    client = TestClient(create_app(analysis_service=_ReadyService()))
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
