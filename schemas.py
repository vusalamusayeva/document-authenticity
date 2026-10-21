from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from typing_extensions import Literal

class BoundingBox(BaseModel):
    ymin: int
    xmin: int
    ymax: int
    xmax: int

class Candidate(BaseModel):
    type: Literal["signature", "stamp"]
    bbox: BoundingBox
    confidence: float
    is_suspicious: bool = False

class StampResult(BaseModel):
    detected: bool
    confidence: float
    analysis_details: str
    circularity: Optional[float] = None
    solidity: Optional[float] = None

class SignatureStampResult(BaseModel):
    file_id: str
    image_width: int
    image_height: int
    candidates: List[Candidate] = Field(default_factory=list)
    signature_verifications: List[Dict[str, Any]] = Field(default_factory=list)
    stamp_analyses: List[Dict[str, Any]] = Field(default_factory=list)
    risk_score: float = Field(0.0, ge=0.0, le=1.0)
    is_suspicious: bool = False
    visualization_path: Optional[str] = None
    status: Literal["success", "failed"] = "success"
    errors: List[str] = Field(default_factory=list)

    @validator('file_id', pre=True, allow_reuse=True)
    def validate_file_id(cls, v):
        if not isinstance(v, str):
            v = str(v)
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Path traversal təhlükəsi: file_id qadağan olunmuş simvollar ehtiva edir.")
        return v