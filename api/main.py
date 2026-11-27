import asyncio
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
import uuid

from api.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse
from api.services.analysis_service import AnalysisService
from doc_auth.api import create_document_router
from doc_auth.deepfake.api import attach_deepfake_router
from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError, VideoValidationError


import shutil
import subprocess as _sp
from typing import Tuple

def check_readiness_state(service_obj: Any, required_langs: Tuple[str, ...] = ("eng", "aze")) -> Tuple[bool, Dict[str, Any]]:
    model_ok = service_obj is not None
    if model_ok and hasattr(service_obj, "is_ready") and callable(service_obj.is_ready):
        try:
            model_ok = bool(service_obj.is_ready())
        except Exception:
            model_ok = False

    tess_bin = shutil.which("tesseract")
    installed_langs: list[str] = []
    ocr_engine = None
    if tess_bin:
        ocr_engine = "tesseract"
        try:
            proc = _sp.run([tess_bin, "--list-langs"], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                installed_langs = [
                    ln.strip() for ln in proc.stdout.splitlines()[1:] if ln.strip()
                ]
        except Exception:
            installed_langs = []
    else:
        try:
            import easyocr  # type: ignore  # noqa: F401
            ocr_engine = "easyocr"
            installed_langs = ["eng", "aze"]
        except Exception:
            ocr_engine = None

    missing_langs = [lg for lg in required_langs if lg not in installed_langs]
    ocr_ok = ocr_engine is not None and len(missing_langs) == 0
    ready_ok = model_ok and ocr_ok
    payload: Dict[str, Any] = {
        "status": "ready" if ready_ok else "not_ready",
        "model_ready": model_ok,
        "ocr_ready": ocr_ok,
        "ocr_engine": ocr_engine,
        "installed_langs": installed_langs,
        "missing_langs": missing_langs,
    }
    return ready_ok, payload


app = FastAPI(title="Document Authenticity API", version="0.1.0")
service = AnalysisService()
attach_deepfake_router(app)
app.include_router(create_document_router())


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    ok, payload = check_readiness_state(service)
    if not ok:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.post("/v1/analyze", response_model=DocumentAnalysisResponse)
async def analyze(request: DocumentAnalysisRequest) -> Any:
    try:
        return await asyncio.to_thread(service.analyze, request)
    except (ValueError, InvalidConfigError, VideoValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Model haz?rda ?l?atan deyil.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Daxili analiz x?tas? ba? verdi.") from exc
