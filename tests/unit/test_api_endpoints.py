from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app
from doc_auth.api import create_app
from doc_auth.api import create_document_router
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


def test_analyze_upload_size_empty_and_magic_byte_errors(tmp_path) -> None:
    import cv2
    import numpy as np

    client = TestClient(app)

    mini_app = FastAPI()
    mini_app.include_router(create_document_router(max_upload_bytes=20))
    mini_client = TestClient(mini_app)

    # 1. Fayl ?l??s? limiti (413)
    r_413 = mini_client.post(
        "/v1/analyze",
        files={"file": ("big.png", b"\x89PNG\r\n\x1a\n" + (b"a" * 50), "image/png")},
    )
    assert r_413.status_code == 413

    # 2. Bo? fayl (400)
    r_empty = mini_client.post(
        "/v1/analyze",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r_empty.status_code == 400

    # 3. Magic byte uzant?ya uy?un deyil (400)
    r_bad_magic = mini_client.post(
        "/v1/analyze",
        files={"file": ("fake.png", b"NOT_A_PNG_HEADER", "image/png")},
    )
    assert r_bad_magic.status_code == 400

    # 4. /v1/analyze/json validasiya v? u?urlu ?a??r??
    r_422 = client.post("/v1/analyze/json", json={})
    assert r_422.status_code == 422

    r_404 = client.post("/v1/analyze/json", json={"file_path": str(tmp_path / "missing.png")})
    assert r_404.status_code == 404

    img_file = tmp_path / "json_doc.png"
    canvas = np.ones((200, 300, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "TEST JSON", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    cv2.imwrite(str(img_file), canvas)

    r_200 = client.post("/v1/analyze/json", json={"file_path": str(img_file)})
    assert r_200.status_code == 200
    assert "checks" in r_200.json()


def test_v1_analyze_single_post_and_openapi_examples() -> None:
    fresh_app = create_app()
    post_analyze_routes = []

    for route in fresh_app.routes:
        if getattr(route, "path", None) == "/v1/analyze" and "POST" in (getattr(route, "methods", None) or set()):
            post_analyze_routes.append(route)
        orig = getattr(route, "original_router", None)
        if orig is not None:
            prefix = getattr(orig, "prefix", "") or ""
            for sub in getattr(orig, "routes", []):
                sub_path = getattr(sub, "path", None)
                full_path = f"{prefix}{sub_path}" if sub_path and not str(sub_path).startswith(prefix) else sub_path
                if full_path == "/v1/analyze" and "POST" in (getattr(sub, "methods", None) or set()):
                    post_analyze_routes.append(sub)

    assert len(post_analyze_routes) == 1

    schema = fresh_app.openapi()
    assert "post" in schema["paths"]["/v1/analyze"]
    assert "200" in schema["paths"]["/v1/analyze"]["post"]["responses"]
    json_post = schema["paths"]["/v1/analyze/json"]["post"]
    examples = json_post["requestBody"]["content"]["application/json"]["examples"]
    assert "file_path_request" in examples
    assert "base64_request" in examples
