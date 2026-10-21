"""FastAPI backend giriş nöqtəsi."""

from fastapi import FastAPI

from api.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse
from api.services.analysis_service import AnalysisService

app = FastAPI(title="Document Authenticity API", version="0.1.0")
service = AnalysisService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=DocumentAnalysisResponse)
def analyze(request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
    return service.analyze(request)
