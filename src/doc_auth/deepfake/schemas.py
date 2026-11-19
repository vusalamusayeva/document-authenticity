from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FacePredictionSchema(BaseModel):
    face_index: int = 0
    bbox: List[int] = Field(default_factory=list)
    score: float
    is_deepfake: bool = False
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None


class AudioAnalysisSchema(BaseModel):
    status: str = "unknown"
    has_audio: Optional[bool] = None
    audio_duration: Optional[float] = None
    video_duration: Optional[float] = None
    duration_diff: Optional[float] = None
    rms_energy: Optional[float] = None
    lip_sync_score: Optional[float] = None
    is_suspicious: Optional[bool] = None


class DeepfakeAnalysisResponse(BaseModel):
    status: str
    is_deepfake: Optional[bool] = None
    risk_score: Optional[float] = None
    max_visual_score: Optional[float] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    module_statuses: Dict[str, str] = Field(default_factory=dict)
    audio: Optional[Dict[str, Any]] = None
    frames_analyzed: int = 0
    faces_detected: int = 0
    face_predictions: List[Dict[str, Any]] = Field(default_factory=list)