from fastapi.testclient import TestClient
from doc_auth.api import create_app
from doc_auth.schemas.pipeline_schemas import DocumentAuthSummary


class _FakeSuccessService:
    def analyze_summary(self, document_path):
        return DocumentAuthSummary(
            request_id="sample_doc.pdf",
            status="completed",
            overall_risk_score=0.12,
            decision="authentic",
            checks=[],
            metadata={"document_path": str(document_path)},
        )


class _FakeFailingService:
    def analyze_summary(self, document_path):
        raise RuntimeError("Pipeline emalı dayandı")


def test_v1_analyze_success():
    client = TestClient(create_app(analysis_service=_FakeSuccessService()))
    response = client.post(
        "/v1/analyze",
        files={"file": ("sample_doc.pdf", b"%PDF-1.4 fake test content", "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "authentic"
    assert data["status"] == "completed"
    assert data["overall_risk_score"] == 0.12


def test_v1_analyze_empty_file_returns_400():
    client = TestClient(create_app(analysis_service=_FakeSuccessService()))
    response = client.post(
        "/v1/analyze",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_v1_analyze_unsupported_extension_returns_400():
    client = TestClient(create_app(analysis_service=_FakeSuccessService()))
    response = client.post(
        "/v1/analyze",
        files={"file": ("script.exe", b"MZ-fake", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_v1_analyze_service_error_returns_500():
    client = TestClient(create_app(analysis_service=_FakeFailingService()))
    response = client.post(
        "/v1/analyze",
        files={"file": ("broken.pdf", b"%PDF-1.4 broken", "application/pdf")},
    )
    assert response.status_code == 500

import cv2
import numpy as np
from doc_auth.service import AnalysisService


def test_v1_analyze_real_service_with_image(tmp_path):
    img_path = tmp_path / "sample_doc.png"
    canvas = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (30, 30), (370, 270), (20, 20, 20), 2)
    cv2.putText(canvas, "OFFICIAL DOCUMENT", (55, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    img_path.write_bytes(encoded.tobytes())

    client = TestClient(create_app(analysis_service=AnalysisService()))
    response = client.post(
        "/v1/analyze",
        files={"file": ("sample_doc.png", img_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"]
    assert "decision" in data
    assert isinstance(data["checks"], list)

def test_health_and_deepfake_router_mounted():
    def _fake_deepfake_runner(_path):
        return {
            "status": "completed",
            "frames_analyzed": 4,
            "faces_detected": 1,
            "decision": "authentic",
        }

    client = TestClient(
        create_app(
            analysis_service=_FakeSuccessService(),
            deepfake_runner=_fake_deepfake_runner,
        )
    )

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}

    deepfake_resp = client.post(
        "/deepfake/analyze",
        files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
    )
    assert deepfake_resp.status_code == 200
    assert deepfake_resp.json()["decision"] == "authentic"