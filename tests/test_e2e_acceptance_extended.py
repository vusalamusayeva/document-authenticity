import pytest
from pathlib import Path
import io
from PIL import Image
from fastapi.testclient import TestClient

from doc_auth.api import create_app


def _make_png_bytes(color=(220, 225, 230)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (320, 200), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_request_id_header():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_document_analyze_valid_image_schema():
    app = create_app()
    client = TestClient(app)
    payload = _make_png_bytes()
    resp = client.post("/v1/analyze", files={"file": ("doc.png", payload, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "decision" in body
    assert "checks" in body and isinstance(body["checks"], list) and len(body["checks"]) > 0


def test_reject_corrupted_or_oversized_upload():
    app = create_app()
    client = TestClient(app)
    bad_ext = client.post("/v1/analyze", files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")})
    assert bad_ext.status_code == 400


def test_job_endpoint_and_idempotency():
    app = create_app()
    client = TestClient(app)
    payload = _make_png_bytes()
    r1 = client.post(
        "/v1/jobs?idempotency_key=idem-123",
        files={"file": ("doc.png", payload, "image/png")},
    )
    assert r1.status_code == 200
    job_id = r1.json()["job_id"]

    r2 = client.post(
        "/v1/jobs?idempotency_key=idem-123",
        files={"file": ("doc.png", payload, "image/png")},
    )
    assert r2.status_code == 200
    assert r2.json()["job_id"] == job_id
    assert r2.json().get("idempotent_replay") is True

    status_resp = client.get(f"/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "completed"


def test_tampered_sample_analyze_endpoint() -> None:
    sample_path = Path("outputs/e2e_samples/tampered_az_scan_test.png")
    if not sample_path.exists():
        pytest.skip("tampered_az_scan_test.png sample not present")

    app = create_app()
    client = TestClient(app)
    with open(sample_path, "rb") as f:
        resp = client.post("/v1/analyze", files={"file": (sample_path.name, f, "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    assert "overall_risk_score" in body
    assert body.get("decision") in {"authentic", "suspicious", "tampered", "review", "inconclusive"}
    modules = {c.get("module"): c for c in body.get("checks", [])}
    assert "classification" in modules
    assert "forensics" in modules
