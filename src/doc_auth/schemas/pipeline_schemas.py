from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class DeepfakeConfig(BaseModel):
    max_file_size_mb: int = Field(default=100)
    max_duration_sec: float = Field(default=60.0)
    max_resolution: tuple = Field(default=(1920, 1080))
    max_frames: int = Field(default=300)
    sampling_rate_fps: float = Field(default=2.0)

class VideoMetadata(BaseModel):
    filename: str
    format: str
    duration_sec: float
    width: int
    height: int
    fps: float
    total_frames: int
    has_audio: bool
    audio_duration_sec: Optional[float] = None

class FrameAnalysisResult(BaseModel):
    frame_index: int
    timestamp_ms: float
    face_detected: bool
    deepfake_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bounding_boxes: List[List[float]] = []

class DeepfakeAnalysisResult(BaseModel):
    status: str = Field(default="success", description="success, partial, failed, model_unavailable")
    metadata: Optional[VideoMetadata] = None
    frames_analyzed: int = 0
    temporal_inconsistency_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    audio_status: Dict[str, Any] = {}
    overall_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_suspicious: bool = False
    error_message: Optional[str] = None