from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    confidence: float = 0.0
    label: Optional[str] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_int_list(self) -> List[int]:
        return [int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))]


class Candidate(BaseModel):
    candidate_id: Optional[str] = None
    element_type: str = "unknown"
    label: Optional[str] = None
    bbox: Any = None
    score: float = 0.0
    confidence: float = 0.0
    is_authentic: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StampResult(BaseModel):
    status: str = "success"
    detected: bool = False
    is_authentic: Optional[bool] = None
    confidence: float = 0.0
    score: Optional[float] = None
    circularity: Optional[float] = None
    solidity: Optional[float] = None
    bbox: Optional[BoundingBox] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class SignatureResult(BaseModel):
    status: str = "success"
    detected: bool = False
    similarity_score: Optional[float] = None
    is_match: Optional[bool] = None
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class TextureAnalysisResult(BaseModel):
    status: str = "success"
    texture_score: float = 0.0
    anomalies_detected: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class SecurityElementsResult(BaseModel):
    status: str = "success"
    elements_found: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class SignatureStampItem(BaseModel):
    element_type: str
    bbox: List[float] = Field(default_factory=list)
    confidence: float = 0.0
    is_authentic: Optional[bool] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SignatureStampResult(BaseModel):
    file_id: str
    status: str = "success"
    overall_score: Optional[float] = None
    risk_score: Optional[float] = None
    is_suspicious: Optional[bool] = None
    is_tampered: Optional[bool] = None
    visualization_path: Optional[str] = None
    signature_verifications: List[Dict[str, Any]] = Field(default_factory=list)
    items: List[SignatureStampItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    @field_validator("file_id", mode="before")
    @classmethod
    def _coerce_file_id(cls, value: Any) -> str:
        if value is None or str(value).strip() == "":
            raise ValueError("file_id boş ola bilməz.")
        return str(value)