from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

AnalysisStatus = Literal["success", "partial", "failed", "model_unavailable"]

class DeepfakeConfig(BaseModel):
    max_file_size_mb: int = 500
    max_duration_sec: int = 300
    allowed_formats: List[str] = [".mp4", ".mov", ".avi", ".webm"]
    sample_rate: int = 5
    max_frames: int = 20
    weights: Dict[str, float] = {"visual": 0.6, "temporal": 0.3, "audio": 0.1}

class VideoMetadata(BaseModel):
    duration: float
    size_mb: float
    width: int
    height: int
    format: str

class BaseAnalysisResult(BaseModel):
    status: AnalysisStatus = "success"
    message: Optional[str] = None

class DeepfakeAnalysisResult(BaseAnalysisResult):
    is_deepfake: Optional[bool] = None
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Dict[Any, Any] = Field(default_factory=dict)

    @field_validator('risk_score', 'confidence', mode='before')
    def py_score_validator(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            return None
        return v

class DocumentAnalysisResponse(BaseAnalysisResult):
    file_id: Optional[str] = None
    deepfake_result: Optional[DeepfakeAnalysisResult] = None
    signature_stamp_result: Optional[Dict[str, Any]] = None
    final_decision: Optional[str] = None
    overall_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)