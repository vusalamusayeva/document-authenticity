import asyncio
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import types

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
import pytest

from doc_auth import api as doc_api
from doc_auth.schemas.pipeline_schemas import DocumentAuthSummary


def _router_client(router) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class _DummyService:
    def __init__(self, mode: str = "summary_obj", ready: bool = True) -> None:
        self.mode = mode
        self.ready = ready

    def is_ready(self) -> bool:
        if self.mode == "ready_raises":
            raise RuntimeError("ready check failed")
        return self.ready

    def analyze_summary(self, path: Path):
        if self.mode == "raise_value":
            raise ValueError("bad content")
        if self.mode == "raise_runtime":
            raise RuntimeError("boom")
        if self.mode == "bad_type":
            return "not-a-summary"
        if self.mode == "dict":
            return {
                "decision": "authentic",
                "risk_score": 0.1,
            }
        return DocumentAuthSummary(
            decision="authentic",
            risk_score=0.05,
        )


def test_readiness_state_and_magic_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert doc_api._validate_magic_bytes(".pdf", b"") is False
    assert doc_api._validate_magic_bytes(".pdf", b"%PDF-1.7") is True
    assert doc_api._validate_magic_bytes(".png", b"\x89PNG\r\n\x1a\nrest") is True
    assert doc_api._validate_magic_bytes(".jpg", b"\xff\xd8\xffdata") is True
    assert doc_api._validate_magic_bytes(".webp", b"RIFF1234WEBP") is True
    assert doc_api._validate_magic_bytes(".tif", b"II*\x00rest") is True
    assert doc_api._validate_magic_bytes(".other", b"any") is True

    monkeypatch.setattr("shutil.which", lambda exe: "C:/fake/tesseract.exe")
    monkeypatch.setattr(
        doc_api._sp,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "List of installed languages\neng\naze\n", ""),
    )
    ok, payload = doc_api.check_readiness_state(_DummyService(ready=True))
    assert ok is True
    assert payload["ocr_engine"] == "tesseract"

    ok_err, _ = doc_api.check_readiness_state(_DummyService(mode="ready_raises"))
    assert ok_err is False

    monkeypatch.setattr("shutil.which", lambda exe: None)
    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=object))
    ok_easy, payload_easy = doc_api.check_readiness_state(_DummyService(ready=True))
    assert ok_easy is True
    assert payload_easy["ocr_engine"] == "easyocr"


def test_document_router_and_app_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _DummyService(mode="dict")
    app = doc_api.create_app(analysis_service=svc, deepfake_runner=lambda p: {"ok": True})
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    monkeypatch.setattr("shutil.which", lambda exe: None)
    monkeypatch.delitem(sys.modules, "easyocr", raising=False)
    assert client.get("/ready").status_code == 503

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    res_ok = client.post("/v1/analyze", files={"file": ("doc.png", png_bytes, "image/png")})
    assert res_ok.status_code == 200
    assert res_ok.json()["decision"] == "authentic"

    c_obj = TestClient(doc_api.create_app(analysis_service=_DummyService(mode="summary_obj")))
    res_obj = c_obj.post("/v1/analyze", files={"file": ("doc.png", png_bytes, "image/png")})
    assert res_obj.status_code == 200
    assert res_obj.json()["decision"] == "authentic"

    router = doc_api.create_document_router(analysis_service=svc)
    analyze_ep = next(r.endpoint for r in router.routes if getattr(r, "path", None) == "/v1/analyze")
    with pytest.raises(HTTPException) as exc_empty:
        asyncio.run(analyze_ep(UploadFile(filename="", file=BytesIO(b"abc"))))
    assert exc_empty.value.status_code == 400

    assert client.post("/v1/analyze", files={"file": ("doc.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/v1/analyze", files={"file": ("doc.png", b"", "image/png")}).status_code == 400
    assert client.post("/v1/analyze", files={"file": ("doc.png", b"NOTPNG", "image/png")}).status_code == 400

    small_client = _router_client(doc_api.create_document_router(analysis_service=svc, max_upload_bytes=10))
    assert small_client.post("/v1/analyze", files={"file": ("doc.png", png_bytes, "image/png")}).status_code == 413

    c_val = _router_client(doc_api.create_document_router(analysis_service=_DummyService(mode="raise_value")))
    assert c_val.post("/v1/analyze", files={"file": ("doc.png", png_bytes, "image/png")}).status_code == 400

    c_rt = _router_client(doc_api.create_document_router(analysis_service=_DummyService(mode="raise_runtime")))
    assert c_rt.post("/v1/analyze", files={"file": ("doc.png", png_bytes, "image/png")}).status_code == 500

    c_bad = _router_client(doc_api.create_document_router(analysis_service=_DummyService(mode="bad_type")))
    assert c_bad.post("/v1/analyze", files={"file": ("doc.png", png_bytes, "image/png")}).status_code == 500

    sample_file = tmp_path / "sample.png"
    sample_file.write_bytes(png_bytes)
    assert client.post("/v1/analyze/json", json={}).status_code == 422
    assert client.post("/v1/analyze/json", json={"file_path": str(tmp_path / "nope.png")}).status_code == 404
    r_json_ok = client.post("/v1/analyze/json", json={"file_path": str(sample_file)})
    assert r_json_ok.status_code == 200
    assert r_json_ok.json()["decision"] == "authentic"

    no_sum_svc = types.SimpleNamespace()
    c_501 = _router_client(doc_api.create_document_router(analysis_service=no_sum_svc))
    assert c_501.post("/v1/analyze/json", json={"file_path": str(sample_file)}).status_code == 501

    assert client.get("/v1/jobs/unknown-id").status_code == 404
    j1 = client.post("/v1/jobs?idempotency_key=k1", files={"file": ("doc.png", png_bytes, "image/png")})
    assert j1.status_code == 200
    job_id = j1.json()["job_id"]
    j2 = client.post("/v1/jobs?idempotency_key=k1", files={"file": ("doc.png", png_bytes, "image/png")})
    assert j2.status_code == 200
    assert j2.json()["idempotent_replay"] is True
    assert client.get(f"/v1/jobs/{job_id}").status_code == 200

    orig_unlink = Path.unlink

    def _flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if "tmp_uploads" in str(self):
            raise OSError("locked")
        orig_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    c_job_err = _router_client(doc_api.create_document_router(analysis_service=_DummyService(mode="raise_runtime")))
    assert c_job_err.post("/v1/jobs", files={"file": ("doc.png", png_bytes, "image/png")}).status_code == 500
    monkeypatch.setattr(Path, "unlink", orig_unlink)
