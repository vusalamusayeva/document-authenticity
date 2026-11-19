from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app
from doc_auth.deepfake.api import attach_deepfake_router


def test_health_and_deepfake_upload_limit() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    mini_app = FastAPI()
    attach_deepfake_router(mini_app, max_upload_bytes=10)
    mini_client = TestClient(mini_app)

    big_payload = b"0" * 50
    over_resp = mini_client.post(
        "/deepfake/analyze",
        files={"file": ("video.mp4", big_payload, "video/mp4")},
    )
    assert over_resp.status_code == 413
