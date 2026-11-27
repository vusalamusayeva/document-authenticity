from typing import Any

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    confidence: float = 0.0
    label: str | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_int_list(self) -> list[int]:
        return [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]


class Candidate(BaseModel):
    candidate_id: str | None = None
    element_type: str = "unknown"
    label: str | None = None
    bbox: Any = None
    score: float = 0.0
    confidence: float = 0.0
    is_authentic: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StampResult(BaseModel):
    status: str = "success"
    detected: bool = False
    is_authentic: bool | None = None
    confidence: float = 0.0
    score: float | None = None
    circularity: float | None = None
    solidity: float | None = None
    bbox: BoundingBox | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SignatureResult(BaseModel):
    status: str = "success"
    detected: bool = False
    similarity_score: float | None = None
    is_match: bool | None = None
    confidence: float = 0.0
    bbox: BoundingBox | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TextureAnalysisResult(BaseModel):
    status: str = "success"
    texture_score: float = 0.0
    anomalies_detected: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class SecurityElementsResult(BaseModel):
    status: str = "success"
    elements_found: list[str] = Field(default_factory=list)
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SignatureStampItem(BaseModel):
    element_type: str
    bbox: list[float] = Field(default_factory=list)
    confidence: float = 0.0
    is_authentic: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SignatureStampResult(BaseModel):
    file_id: str
    status: str = "success"
    overall_score: float | None = None
    risk_score: float | None = None
    is_suspicious: bool | None = None
    is_tampered: bool | None = None
    visualization_path: str | None = None
    signature_verifications: list[dict[str, Any]] = Field(default_factory=list)
    items: list[SignatureStampItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("file_id", mode="before")
    @classmethod
    def _coerce_file_id(cls, value: Any) -> str:
        if value is None or str(value).strip() == "":
            raise ValueError("file_id boş ola bilməz.")
        return str(value)