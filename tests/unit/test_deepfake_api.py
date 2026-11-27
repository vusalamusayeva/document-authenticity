import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from doc_auth.deepfake import api as df_api
from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError, VideoValidationError


class _DummyDetector:
    def __init__(self, model_path, threshold=0.5, expected_sha256=None) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.expected_sha256 = expected_sha256

    def load_model(self):
        return {"loaded": True, "backend": "dummy"}

    @staticmethod
    def compute_file_sha256(path: Path) -> str:
        return "abc123sha256"


class _DummyPipeline:
    def __init__(self, visual_detector=None) -> None:
        self.visual_detector = visual_detector

    def run(self, path):
        return {"status": "success", "path": str(path), "score": 0.12}


def test_resolve_and_build_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "weights.npz"
    weights.write_bytes(b"model-weights")

    monkeypatch.setenv("DEEPFAKE_MODEL_SHA256", "  custom_sha  ")
    monkeypatch.setenv("DEEPFAKE_THRESHOLD", "0.75")
    p_str, sha_str, thr = df_api._resolve_model_config()
    assert sha_str == "custom_sha"
    assert thr == pytest.approx(0.75)
    assert isinstance(p_str, str)

    monkeypatch.setenv("DEEPFAKE_MODEL_SHA256", "   ")
    _, sha_empty, _ = df_api._resolve_model_config(expected_sha256=" explicit_sha ")
    assert sha_empty == "explicit_sha"

    _, sha_none, _ = df_api._resolve_model_config(expected_sha256="")
    assert sha_none is None

    monkeypatch.setattr(df_api, "VisualDetector", _DummyDetector)
    monkeypatch.setattr(df_api, "DeepfakePipeline", _DummyPipeline)
    pipe, info = df_api.build_deepfake_pipeline_from_config(
        model_path=weights,
        expected_sha256="sha_val",
        threshold=0.65,
    )
    assert isinstance(pipe, _DummyPipeline)
    assert info["sha256"] == "abc123sha256"
    assert info["threshold"] == pytest.approx(0.65)


def test_router_unready_and_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("missing weights")

    monkeypatch.setattr(df_api, "build_deepfake_pipeline_from_config", _boom)

    app_unready = FastAPI()
    df_api.attach_deepfake_router(app_unready)
    client_unready = TestClient(app_unready)

    r_ready = client_unready.get("/deepfake/readiness")
    assert r_ready.status_code == 503

    r_analyze_503 = client_unready.post(
        "/deepfake/analyze",
        files={"file": ("clip.mp4", b"data", "video/mp4")},
    )
    assert r_analyze_503.status_code == 503

    # Test _default_runner directly when state is not ready
    router_unready = df_api.create_deepfake_router()
    analyze_fn = None
    for route in router_unready.routes:
        if getattr(route, "path", None) == "/deepfake/analyze":
            analyze_fn = route.endpoint
            break
    assert analyze_fn is not None


def test_analyze_endpoint_status_codes_and_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()

    # 1. Empty filename -> 400
    router_400 = df_api.create_deepfake_router(pipeline_runner=lambda p: {"ok": True})
    app.include_router(router_400)
    client = TestClient(app)

    res_empty_name = client.post("/deepfake/analyze", files={"file": ("", b"123", "video/mp4")})
    assert res_empty_name.status_code == 400

    # 2. Empty file content -> 400
    res_empty_bytes = client.post("/deepfake/analyze", files={"file": ("a.mp4", b"", "video/mp4")})
    assert res_empty_bytes.status_code == 400

    # 3. Exceeding max_upload_bytes -> 413
    small_router = df_api.create_deepfake_router(pipeline_runner=lambda p: {"ok": True}, max_upload_bytes=4)
    app_small = FastAPI()
    app_small.include_router(small_router)
    client_small = TestClient(app_small)
    res_413 = client_small.post("/deepfake/analyze", files={"file": ("a.mp4", b"12345", "video/mp4")})
    assert res_413.status_code == 413

    # 4. Non-dict result -> wrapped in {"status": "success", "result": ...}
    app_list = FastAPI()
    df_api.attach_deepfake_router(app_list, pipeline_runner=lambda p: ["frame1"])
    client_list = TestClient(app_list)
    res_list = client_list.post("/deepfake/analyze", files={"file": ("a.mp4", b"abc", "video/mp4")})
    assert res_list.status_code == 200
    assert res_list.json() == {"status": "success", "result": ["frame1"]}

    # 5. Result with status == "model_unavailable" -> 503
    app_unavail_dict = FastAPI()
    df_api.attach_deepfake_router(
        app_unavail_dict,
        pipeline_runner=lambda p: {"status": "model_unavailable", "error": "offline"},
    )
    res_unavail_dict = TestClient(app_unavail_dict).post(
        "/deepfake/analyze", files={"file": ("a.mp4", b"abc", "video/mp4")}
    )
    assert res_unavail_dict.status_code == 503

    # 6. ValidationError -> 422
    app_422 = FastAPI()
    df_api.attach_deepfake_router(
        app_422,
        pipeline_runner=lambda p: (_ for _ in ()).throw(VideoValidationError("bad video")),
    )
    assert TestClient(app_422).post("/deepfake/analyze", files={"file": ("a.mp4", b"abc", "video/mp4")}).status_code == 422

    # 7. ModelUnavailableError raised in runner -> 503
    app_503_exc = FastAPI()
    df_api.attach_deepfake_router(
        app_503_exc,
        pipeline_runner=lambda p: (_ for _ in ()).throw(ModelUnavailableError("no model")),
    )
    assert TestClient(app_503_exc).post("/deepfake/analyze", files={"file": ("a.mp4", b"abc", "video/mp4")}).status_code == 503

    # 8. Unexpected exception -> 500 + unlink OSError handled silently
    orig_unlink = Path.unlink

    def _flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if "tmp" in self.name.lower() or self.suffix == ".mp4":
            raise OSError("locked file")
        orig_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    app_500 = FastAPI()
    df_api.attach_deepfake_router(
        app_500,
        pipeline_runner=lambda p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert TestClient(app_500).post("/deepfake/analyze", files={"file": ("a.mp4", b"abc", "video/mp4")}).status_code == 500
    monkeypatch.setattr(Path, "unlink", orig_unlink)
