from fastapi.testclient import TestClient

from doc_auth.api import app


def test_api_health_and_ready_endpoints() -> None:
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() is not None

    ready_resp = client.get("/ready")
    assert ready_resp.status_code in (200, 503)
    assert ready_resp.json() is not None
