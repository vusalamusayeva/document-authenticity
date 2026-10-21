"""JSON input/output strukturları (Pydantic)."""

from pydantic import BaseModel, Field


class DocumentAnalysisRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded giriş şəkli")
    document_type_hint: str | None = None


class RegionScore(BaseModel):
    bbox: list[float]  # [x1, y1, x2, y2]
    tamper_type: str
    score: float


class DocumentAnalysisResponse(BaseModel):
    label: str  # authentic | suspicious | forged
    confidence: float
    document_type: str | None = None
    suspicious_regions: list[RegionScore] = []
    explanation_mask_base64: str | None = None
    request_id: str
