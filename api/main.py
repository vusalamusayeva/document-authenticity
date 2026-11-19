import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException

from api.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse
from api.services.analysis_service import AnalysisService
from doc_auth.deepfake.api import attach_deepfake_router
from doc_auth.deepfake.exceptions import InvalidConfigError, ModelUnavailableError, VideoValidationError

app = FastAPI(title="Document Authenticity API", version="0.1.0")
service = AnalysisService()
attach_deepfake_router(app)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=DocumentAnalysisResponse)
async def analyze(request: DocumentAnalysisRequest) -> Any:
    try:
        return await asyncio.to_thread(service.analyze, request)
    except (ValueError, InvalidConfigError, VideoValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Model əlçatan deyil: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Daxili analiz xətası: {exc}") from exc