import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Request
import uuid

from api.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse
from api.services.analysis_service import AnalysisService
from doc_auth.api import create_document_router
from doc_auth.deepfake.api import attach_deepfake_router
from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError, VideoValidationError

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
    return {"status": "ready", "service": "ok"}


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
